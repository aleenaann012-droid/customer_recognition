import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import threading
import time
import queue
import os
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
conn   = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT name, role, embedding FROM staff")

staff_details    = []
staff_embeddings = []
for name, role, embedding_blob in cursor.fetchall():
    staff_details.append({"name": name, "role": role})
    staff_embeddings.append(np.frombuffer(embedding_blob, dtype=np.float32))

conn.close()

# ==========================================
# SHARED STATE
# ==========================================

frame_queue = queue.Queue(maxsize=1)
det_queue   = queue.Queue(maxsize=1)

draw_results  = []   # list of {box, text} for the main loop to render
results_lock  = threading.Lock()

active_tracks = {}   # track_id → {box, text, last_seen}
next_track_id = 0

# ==========================================
# DETECTION THREAD
# — Only runs InsightFace, no matching logic
# ==========================================

def detection_thread():
    while True:
        try:
            frame = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        faces = app.get(frame)

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
                sim = cosine_similarity(embedding, known_emb.reshape(1, -1))[0][0]
                if sim > 0.6:
                    display_name = f"{staff_details[i]['name']} | {staff_details[i]['role']}"
                    break

            text = f"{display_name} | {gender} | Age: {age}"


            active_tracks[best_track_id] = {
                "box":       final_box,
                "text":      text,
                "last_seen": current_time,
            }

        for tid in [tid for tid, d in active_tracks.items()
                    if current_time - d["last_seen"] > TRACK_TIMEOUT]:
            del active_tracks[tid]

        temp = [{"box": d["box"], "text": d["text"]}
                for d in active_tracks.values()]

        with results_lock:
            draw_results = temp

# ==========================================
# START THREADS
# ==========================================

threading.Thread(target=detection_thread,   daemon=True).start()
threading.Thread(target=recognition_thread, daemon=True).start()


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    try:
        if frame_queue.full():
            frame_queue.get_nowait()
        frame_queue.put_nowait(frame.copy())
    except queue.Full:
        pass

    display = frame.copy()


    with results_lock:
        for res in draw_results:
            box  = res["box"]
            text = res["text"]

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
                (box[0], box[1] - 10),
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