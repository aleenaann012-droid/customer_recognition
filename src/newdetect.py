from cv2 import text
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import threading
import time
import queue
import os
import supervision as sv
import warnings

# Suppress the FutureWarning from supervision about ByteTrack deprecation
warnings.filterwarnings("ignore", category=FutureWarning, module="supervision")
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

frame_counter = 0
staff_presence = {}
recognized_tracks = {}
saved_tracks = set()
tracker = sv.ByteTrack(
    track_activation_threshold=0.25, 
    minimum_consecutive_frames=1, 
    minimum_matching_threshold=0.5
)
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
        boxes=[]
        scores=[]
        for face in faces:
            boxes.append(face.bbox)
            scores.append(face.det_score)
            
        if len(boxes) > 0:
            detections = sv.Detections(
                xyxy=np.array(boxes),
                confidence=np.array(scores),
                class_id=np.zeros(len(boxes), dtype=int)
            )
        else:
            detections = sv.Detections.empty()
            
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
            det_queue.put_nowait(
    (faces, detections)
)
        except queue.Full:
            pass

def recognition_thread():
    global draw_results

    while True:
        try:
            faces, detections = det_queue.get(timeout=1.0)
            tracks = tracker.update_with_detections(detections)
            print("Tracks length:", len(tracks))

            if len(tracks) > 0:
                print("Tracker IDs:", tracks.tracker_id)
                for t_box, t_id in zip(tracks.xyxy, tracks.tracker_id):
                    print(f"  Box: {t_box}, Track ID: {t_id}")
        except queue.Empty:
            continue

        temp_results = []
        for i_face, face in enumerate(faces):
            matched_tid = i_face
            best_dist = 9999
            fx_c = (face.bbox[0] + face.bbox[2]) / 2
            fy_c = (face.bbox[1] + face.bbox[3]) / 2
            
            if len(tracks) > 0:
                for t_box, t_id in zip(tracks.xyxy, tracks.tracker_id):
                    tx_c = (t_box[0] + t_box[2]) / 2
                    ty_c = (t_box[1] + t_box[3]) / 2
                    dist = ((fx_c - tx_c)**2 + (fy_c - ty_c)**2)**0.5
                    if dist < 50 and dist < best_dist:
                        best_dist = dist
                        matched_tid = int(t_id)
            
            embedding    = face.embedding.reshape(1, -1)
            age          = int(face.age)
            gender       = "Male" if face.gender == 1 else "Female"
            display_name = "CUSTOMER"

            for i, known_emb in enumerate(staff_embeddings):

                sim = cosine_similarity(
                        embedding,
                        known_emb.reshape(1, -1)
                    )[0][0]

                if sim > 0.3:

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
            text = f"ID:{matched_tid} | {display_name} | {gender} | Age:{age}"
            
            if display_name == "CUSTOMER" and matched_tid not in saved_tracks:
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
                    saved_tracks.add(matched_tid)
                except Exception as e:
                    print(f"Error inserting visitor stats: {e}")
            
            temp_results.append({
                "track_id": matched_tid,
                "box": face.bbox,
                "text": text
            })

        with results_lock:
            global new_recognitions_flag
            draw_results = temp_results
            new_recognitions_flag = True

threading.Thread(target=detection_thread,   daemon=True).start()
threading.Thread(target=recognition_thread, daemon=True).start()


cap = cv2.VideoCapture(r"C:\Users\USER\Downloads\test.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(
    'tracked_output.mp4',
    fourcc,
    25,
    (640, 480)
)
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
        current_results = draw_results.copy()

    for result in current_results:
        box = result["box"]
        text = result["text"]

        cv2.rectangle(
            display,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            text,
            (int(box[0]), max(10, int(box[1]) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.putText(
    display,
    "VIDEO RECORDING TEST",
    (50,50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0,0,255),
    2
)
    out.write(display)
    cv2.imshow("Recognition System", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

out.release()
cap.release()
cv2.destroyAllWindows()