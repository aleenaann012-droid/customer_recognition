import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import threading
import time
import queue
import os
from tracker import create_tracker
def get_age_group(age):

    if age <= 17:
        return "0-17"

    elif age <= 25:
        return "18-25"

    elif age <= 35:
        return "26-35"

    elif age <= 45:
        return "36-45"

    elif age <= 60:
        return "46-60"

    else:
        return "60+"
# ==========================================
# CONFIGURATION
# ==========================================

SMOOTH_ALPHA   = 0.8  # 0.0 = smoothest box movement, 1.0 = instant snap
MAX_MATCH_DIST = 100   
TRACK_TIMEOUT  = 0.5


# ==========================================
# LOAD MODELS & DATABASE
# ==========================================

app = FaceAnalysis()
app.prepare(ctx_id=-1)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "..", "database", "customers.db")
conn   = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()
cursor.execute(
    "SELECT staff_id, name, role, embedding FROM staff"
)
staff_details = []
staff_embeddings = []

for staff_id, name, role, embedding_blob in cursor.fetchall():

    staff_details.append({
        "staff_id": staff_id,
        "name": name,
        "role": role
    })

    staff_embeddings.append(
        np.frombuffer(
            embedding_blob,
            dtype=np.float32
        )
    )

# # ==========================
# # LOAD CUSTOMERS
# # ==========================

# customer_embeddings = []

# cursor.execute("""
# SELECT customer_id,
#        embedding,
#        visit_count
# FROM customers
# """)

# for customer_id, emb_blob, visits in cursor.fetchall():

#     customer_embeddings.append({

#         "id": customer_id,

#         "embedding": np.frombuffer(
#             emb_blob,
#             dtype=np.float32
#         ),

#         "visits": visits
#     })
# print("Staff:", len(staff_embeddings))
# print("Customers:", len(customer_embeddings))   


# ==========================================
# SHARED STATE
# ==========================================

frame_queue = queue.Queue(maxsize=1)
det_queue   = queue.Queue(maxsize=1)

draw_results  = []
results_lock  = threading.Lock()
new_recognitions_flag = False

active_tracks = {}
next_track_id = 0
frame_counter = 0
saved_tracks = set()
staff_presence = {}

# ==========================================
# DETECTION THREAD
# — Only runs InsightFace, no matching logic
# ==========================================
 
def detection_thread():
    global frame_counter

    while True:

        try:
            frame = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        frame_counter += 1

        if frame_counter % 3 != 0:
            continue

        small_frame = cv2.resize(frame, (320, 240))

        faces = app.get(small_frame)

        scale_x = frame.shape[1] / 320
        scale_y = frame.shape[0] / 240

        for face in faces:
            face.bbox[0] *= scale_x
            face.bbox[1] *= scale_y
            face.bbox[2] *= scale_x
            face.bbox[3] *= scale_y

        # Always push the latest result, drop stale one if needed
        try:
            if det_queue.full():
                det_queue.get_nowait()
            det_queue.put_nowait(faces)
        except queue.Full:
            pass

# ==========================================
# RECOGNITION THREAD
# ==========================================

def recognition_thread():
    global draw_results, active_tracks, next_track_id

    while True:
        try:
            faces = det_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        current_time   = time.time()
        matched_tracks = set()

        for face in faces:
            new_box    = face.bbox.astype(int)
            new_center = (
                (new_box[0] + new_box[2]) / 2,
                (new_box[1] + new_box[3]) / 2,
            )

            best_track_id = -1
            min_dist      = MAX_MATCH_DIST

            for track_id, track_data in active_tracks.items():
                if track_id in matched_tracks:
                    continue

                old_box    = track_data["box"]
                old_center = (
                    (old_box[0] + old_box[2]) / 2,
                    (old_box[1] + old_box[3]) / 2,
                )

                # FIX: **2 (power), not *2 (multiply)
                dist = np.sqrt(
                    (new_center[0] - old_center[0]) ** 2 +
                    (new_center[1] - old_center[1]) ** 2
                )

                if dist < min_dist:
                    min_dist      = dist
                    best_track_id = track_id

            if best_track_id != -1:
                old_box   = active_tracks[best_track_id]["box"]
                final_box = [
                    int(old_box[0] * (1 - SMOOTH_ALPHA) + new_box[0] * SMOOTH_ALPHA),
                    int(old_box[1] * (1 - SMOOTH_ALPHA) + new_box[1] * SMOOTH_ALPHA),
                    int(old_box[2] * (1 - SMOOTH_ALPHA) + new_box[2] * SMOOTH_ALPHA),
                    int(old_box[3] * (1 - SMOOTH_ALPHA) + new_box[3] * SMOOTH_ALPHA),
                ]
                matched_tracks.add(best_track_id)
            else:
                final_box     = new_box.tolist()
                best_track_id = next_track_id
                next_track_id += 1
                matched_tracks.add(best_track_id)

            embedding    = face.embedding.reshape(1, -1)
            age          = int(face.age)
            gender       = "Male" if face.gender == 1 else "Female"
            display_name = "CUSTOMER"

            for i, known_emb in enumerate(staff_embeddings):

                sim = cosine_similarity(
                    embedding,
                    known_emb.reshape(1, -1)
                )[0][0]

                if sim > 0.6:

                    staff_name = staff_details[i]["name"]

                    display_name = (
                        f"{staff_details[i]['name']} | "
                        f"{staff_details[i]['role']}"
                    )

                    if staff_name not in staff_presence:
                        try:
                            local_cursor = conn.cursor()
                            local_cursor.execute(
                                """
                                INSERT INTO staff_attendance
                                (staff_id, entry_time, date)
                                VALUES
                                (?, datetime('now'), date('now'))
                                """,
                                (staff_details[i]["staff_id"],)
                            )
                            conn.commit()
                            staff_presence[staff_name] = True
                        except Exception as e:
                            print(f"Error inserting staff attendance: {e}")

                    break

            text = f"{display_name} | {gender} | Age: {age}"
            if best_track_id not in saved_tracks:

                age_group = get_age_group(age)

                try:
                    local_cursor = conn.cursor()
                    local_cursor.execute(
                            """
                            INSERT INTO visitor_stats
                            (age_group, gender)
                            VALUES (?, ?)
                            """,
                            (age_group, gender)
                        )
                    conn.commit()
                except Exception as e:
                    print(f"Error inserting visitor stats: {e}")

                saved_tracks.add(best_track_id)

            active_tracks[best_track_id] = {
                "box":       final_box,
                "text":      text,
                "last_seen": current_time,
            }

        for tid in [tid for tid, d in active_tracks.items()
                    if current_time - d["last_seen"] > TRACK_TIMEOUT]:

            track_text = active_tracks[tid]["text"]

            for staff in staff_details:
                if staff["name"] in track_text:
                    try:
                        local_cursor = conn.cursor()
                        local_cursor.execute(
                            """
                            UPDATE staff_attendance
                            SET exit_time = datetime('now')
                            WHERE staff_id = ?
                            AND exit_time IS NULL
                            """,
                            (staff["staff_id"],)
                        )
                        conn.commit()

                        if staff["name"] in staff_presence:
                            del staff_presence[staff["name"]]
                    except Exception as e:
                        print(f"Error updating staff exit time: {e}")

            del active_tracks[tid]

        temp = [{"track_id": tid, "box": d["box"], "text": d["text"]}
                for tid, d in active_tracks.items()]

        with results_lock:
            global draw_results, new_recognitions_flag
            draw_results = temp
            new_recognitions_flag = True

# ==========================================
# START THREADS
# ==========================================

threading.Thread(target=detection_thread,   daemon=True).start()
threading.Thread(target=recognition_thread, daemon=True).start()


cap = cv2.VideoCapture(0)
trackers = {} # track_id -> {tracker, text, box}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    display = frame.copy()

    try:
        if frame_queue.full():
            frame_queue.get_nowait()
        frame_queue.put_nowait(frame.copy())
    except queue.Full:
        pass

    with results_lock:
        if new_recognitions_flag:
            current_recognitions = draw_results.copy()
            new_recognitions_flag = False
        else:
            current_recognitions = None

    if current_recognitions is not None:
        new_trackers = {}
        for rec in current_recognitions:
            tid = rec["track_id"]
            box = rec["box"]
            text = rec["text"]
            
            tracker = create_tracker()
            x1, y1, x2, y2 = box
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            
            if w > 0 and h > 0:
                tracker.init(display, (x1, y1, w, h))
                new_trackers[tid] = {"tracker": tracker, "text": text, "box": box}
        trackers = new_trackers
    else:
        for tid, data in trackers.items():
            ok, bbox = data["tracker"].update(display)
            if ok:
                x, y, w, h = [int(v) for v in bbox]
                data["box"] = [x, y, x + w, y + h]

    for tid, data in trackers.items():
        box  = data["box"]
        text = data["text"]

        cv2.rectangle(
            display,
            (box[0], box[1]),
            (box[2], box[3]),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            display,
            text,
            (box[0], max(10, box[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Recognition System", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()