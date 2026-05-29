import cv2
import numpy as np
from insightface.app import FaceAnalysis
import threading
import queue

# ==========================================
# CONFIGURATION
# ==========================================

SMOOTH_ALPHA   = 0.3
MAX_MATCH_DIST = 200
TRACK_TIMEOUT  = 2.0
MAX_WIDTH      = 640

# ==========================================
# LOAD MODEL
# ==========================================

app = FaceAnalysis(allowed_modules=["detection"])
app.prepare(ctx_id=-1)

# ==========================================
# HELPER
# ==========================================

def resize_keep_aspect(frame, max_width=MAX_WIDTH):
    h, w = frame.shape[:2]
    scale = max_width / w
    return cv2.resize(frame, (max_width, int(h * scale)))

# ==========================================
# SHARED STATE
# ==========================================

frame_queue   = queue.Queue(maxsize=1)
faces_queue   = queue.Queue(maxsize=1)
draw_results  = []
results_lock  = threading.Lock()
active_tracks = {}
next_track_id = 0

# ==========================================
# DETECTION THREAD
# ==========================================

def detection_thread():
    while True:
        try:
            frame = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        faces = app.get(frame)

        try:
            if faces_queue.full():
                faces_queue.get_nowait()
            faces_queue.put_nowait(faces)
        except queue.Full:
            pass

# ==========================================
# TRACKING THREAD
# ==========================================

def tracking_thread():
    global draw_results, active_tracks, next_track_id
    import time

    while True:
        try:
            faces = faces_queue.get(timeout=1.0)
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

            for track_id, data in active_tracks.items():
                if track_id in matched_tracks:
                    continue
                old_box    = data["box"]
                old_center = (
                    (old_box[0] + old_box[2]) / 2,
                    (old_box[1] + old_box[3]) / 2,
                )
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

            active_tracks[best_track_id] = {
                "box":       final_box,
                "last_seen": current_time,
            }

        # Expire stale tracks
        for tid in [tid for tid, d in active_tracks.items()
                    if current_time - d["last_seen"] > TRACK_TIMEOUT]:
            del active_tracks[tid]

        temp = [{"box": d["box"]} for d in active_tracks.values()]

        with results_lock:
            draw_results = temp

# ==========================================
# START THREADS
# ==========================================

threading.Thread(target=detection_thread, daemon=True).start()
threading.Thread(target=tracking_thread,  daemon=True).start()

# ==========================================
# MAIN LOOP
# ==========================================

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = resize_keep_aspect(frame)

    try:
        if frame_queue.full():
            frame_queue.get_nowait()
        frame_queue.put_nowait(frame.copy())
    except queue.Full:
        pass

    display = frame.copy()

    with results_lock:
        for res in draw_results:
            box = res["box"]
            cv2.rectangle(
                display,
                (box[0], box[1]),
                (box[2], box[3]),
                (0, 255, 0),
                2,
            )

    cv2.imshow("Face Detection", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
