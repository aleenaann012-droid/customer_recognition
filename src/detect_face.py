import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import threading
import time
import queue

# Load InsightFace
app = FaceAnalysis()
app.prepare(ctx_id=-1)

# ==========================================
# SHARED VARIABLES & QUEUES
# ==========================================

frame_queue = queue.Queue(maxsize=1)
det_queue = queue.Queue(maxsize=1)

draw_results = []
results_lock = threading.Lock()
last_det_time = 0

conn = sqlite3.connect("database/customers.db")
cursor = conn.cursor()

# Load staff data
cursor.execute("SELECT name, role, embedding FROM staff")
staff_records = cursor.fetchall()
staff_details = []
staff_embeddings = []
for name, role, embedding_blob in staff_records:
    staff_details.append({"name": name, "role": role})
    staff_embeddings.append(np.frombuffer(embedding_blob, dtype=np.float32))

# Open webcam
cap = cv2.VideoCapture(0)

# ==========================================
# DETECTION THREAD
# ==========================================

def detection_thread():
    while True:
        try:
            current_frame = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        # Run face detection and feature extraction
        faces = app.get(current_frame)

        try:
            if det_queue.full():
                det_queue.get_nowait()
            det_queue.put_nowait(faces)
        except queue.Full:
            pass

# ==========================================
# TRACKING & SMOOTHING PARAMETERS
# ==========================================

SMOOTH_ALPHA = 0.5  # 0.0 to 1.0 (lower is smoother, higher is more responsive)
active_tracks = {}
next_track_id = 0

# ==========================================
# RECOGNITION THREAD
# ==========================================

def recognition_thread():
    global draw_results, last_det_time, active_tracks, next_track_id

    while True:
        try:
            faces = det_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        current_time = time.time()
        matched_tracks = set()

        for face in faces:
            new_box = face.bbox.astype(int)
            new_center = ((new_box[0] + new_box[2]) / 2, (new_box[1] + new_box[3]) / 2)
            
            best_track_id = -1
            min_dist = 100  # max pixel distance to link same face
            
            for track_id, track_data in active_tracks.items():
                if track_id in matched_tracks:
                    continue
                old_box = track_data["box"]
                old_center = ((old_box[0] + old_box[2]) / 2, (old_box[1] + old_box[3]) / 2)
                dist = np.sqrt((new_center[0] - old_center[0])**2 + (new_center[1] - old_center[1])**2)
                
                if dist < min_dist:
                    min_dist = dist
                    best_track_id = track_id
                    
            if best_track_id != -1:
                # Smooth box
                old_box = active_tracks[best_track_id]["box"]
                final_box = [
                    int(old_box[0] * (1 - SMOOTH_ALPHA) + new_box[0] * SMOOTH_ALPHA),
                    int(old_box[1] * (1 - SMOOTH_ALPHA) + new_box[1] * SMOOTH_ALPHA),
                    int(old_box[2] * (1 - SMOOTH_ALPHA) + new_box[2] * SMOOTH_ALPHA),
                    int(old_box[3] * (1 - SMOOTH_ALPHA) + new_box[3] * SMOOTH_ALPHA)
                ]
                matched_tracks.add(best_track_id)
            else:
                final_box = new_box.tolist() if isinstance(new_box, np.ndarray) else new_box
                best_track_id = next_track_id
                next_track_id += 1
                matched_tracks.add(best_track_id)
            
            # Feature extraction & matching
            embedding = face.embedding.reshape(1, -1)
            age = int(face.age)
            gender = "Male" if face.gender == 1 else "Female"

            display_name = "CUSTOMER"

            for i, known_embedding in enumerate(staff_embeddings):
                similarity = cosine_similarity(
                    embedding,
                    known_embedding.reshape(1, -1)
                )[0][0]

                if similarity > 0.6:
                    display_name = f"{staff_details[i]['name']} | {staff_details[i]['role']}"
                    break

            text = f"{display_name} | {gender} | Age:{age}"

            # Update track
            active_tracks[best_track_id] = {
                "box": final_box,
                "text": text,
                "last_seen": current_time
            }

        # Remove stale tracks (older than 1.0 second)
        tracks_to_delete = [
            tid for tid, data in active_tracks.items() 
            if current_time - data["last_seen"] > 1.0
        ]
        for tid in tracks_to_delete:
            del active_tracks[tid]

        # Prepare results for drawing
        temp_results = []
        for data in active_tracks.values():
            temp_results.append({
                "box": data["box"],
                "text": data["text"]
            })

        # Update the shared results safely
        with results_lock:
            draw_results = temp_results
            last_det_time = current_time

# ==========================================
# START THREADS
# ==========================================

thread_det = threading.Thread(target=detection_thread)
thread_det.daemon = True
thread_det.start()

thread_rec = threading.Thread(target=recognition_thread)
thread_rec.daemon = True
thread_rec.start()

# ==========================================
# MAIN LOOP (UI & Camera)
# ==========================================

while True:
    ret, captured_frame = cap.read()
    if not ret:
        break

    captured_frame = cv2.resize(captured_frame, (640, 480))
    
    # Send to detection thread (drop old frame if queue is full)
    try:
        if frame_queue.full():
            frame_queue.get_nowait()
        frame_queue.put_nowait(captured_frame.copy())
    except queue.Full:
        pass

    display_frame = captured_frame.copy()
    
    # Draw the latest available results if they are not stale (e.g. > 0.5 seconds old)
    with results_lock:
        if time.time() - last_det_time < 0.5:
            for res in draw_results:
                box = res["box"]
                text = res["text"]
                
                cv2.rectangle(
                    display_frame,
                    (box[0], box[1]),
                    (box[2], box[3]),
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    display_frame,
                    text,
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

    cv2.imshow("Recognition System", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
