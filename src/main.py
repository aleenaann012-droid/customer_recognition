import cv2
import numpy as np
from insightface.app import FaceAnalysis
import threading
import queue
import sqlite3
import time
import pickle

# ==========================================
# CONFIGURATION
# ==========================================

SMOOTH_ALPHA      = 0.3
MAX_MATCH_DIST    = 200          # px — tracker association threshold
TRACK_TIMEOUT     = 2.0          # seconds before a lost track is dropped
MAX_WIDTH         = 640
DB_PATH           = "D:\ai_project\database\customers.db"
RECOG_THRESHOLD   = 0.45         # cosine similarity; above = recognised
RECOG_EVERY_N     = 5            # run recognition on every Nth detection batch
EMBED_RELOAD_SEC  = 30           # reload DB embeddings this often (live updates)

# ==========================================
# LOAD MODEL  (detection + recognition)
# ==========================================

app = FaceAnalysis(allowed_modules=["detection", "recognition"])
app.prepare(ctx_id=-1)          # ctx_id=0 for GPU, -1 for CPU

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def resize_keep_aspect(frame, max_width=MAX_WIDTH):
    h, w = frame.shape[:2]
    scale = max_width / w
    return cv2.resize(frame, (max_width, int(h * scale)))


def cosine_similarity(a, b):
    """Normalised dot product; range [–1, 1], higher = more similar."""
    a = a / (np.linalg.norm(a) + 1e-6)
    b = b / (np.linalg.norm(b) + 1e-6)
    return float(np.dot(a, b))


def load_embeddings_from_db(db_path):
    """
    Returns a list of dicts:
        [{"staff_id": int, "name": str, "role": str, "embedding": np.ndarray}, ...]

    Handles embeddings stored as raw float32 bytes OR pickle blobs.
    """
    records = []
    try:
        conn = sqlite3.connect(db_path)
        cur  = conn.cursor()
        cur.execute("SELECT staff_id, name, role, embedding FROM staff")
        rows = cur.fetchall()
        conn.close()

        for staff_id, name, role, blob in rows:
            if blob is None:
                continue
            try:
                # Try raw float32 first (InsightFace default)
                emb = np.frombuffer(blob, dtype=np.float32).copy()
                if emb.ndim != 1 or emb.size == 0:
                    raise ValueError("bad shape")
            except Exception:
                # Fall back to pickle
                emb = pickle.loads(blob)
                emb = np.array(emb, dtype=np.float32)

            emb = emb / (np.linalg.norm(emb) + 1e-6)   # pre-normalise
            records.append({
                "staff_id":  staff_id,
                "name":      name,
                "role":      role,
                "embedding": emb,
            })

        print(f"[DB] Loaded {len(records)} staff embedding(s).")
    except Exception as e:
        print(f"[DB] Error loading embeddings: {e}")

    return records


def match_embedding(query_emb, db_records, threshold=RECOG_THRESHOLD):
    """
    Compare query embedding against every DB record.
    Returns (name, role, similarity) for the best match above threshold,
    or ("Unknown", "", best_sim) otherwise.
    """
    if not db_records or query_emb is None:
        return "Unknown", "", 0.0

    query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-6)

    best_sim  = -1.0
    best_rec  = None

    for rec in db_records:
        sim = cosine_similarity(query_emb, rec["embedding"])
        if sim > best_sim:
            best_sim = sim
            best_rec = rec

    if best_sim >= threshold and best_rec is not None:
        return best_rec["name"], best_rec["role"], best_sim
    return "Unknown", "", best_sim

# ==========================================
# SHARED STATE
# ==========================================

frame_queue   = queue.Queue(maxsize=1)
faces_queue   = queue.Queue(maxsize=1)
draw_results  = []
results_lock  = threading.Lock()

active_tracks = {}      # track_id -> {box, last_seen, name, role, sim}
next_track_id = 0
track_lock    = threading.Lock()

db_records       = []
db_records_lock  = threading.Lock()

# ==========================================
# DB RELOAD THREAD  (keeps embeddings fresh)
# ==========================================

def db_reload_thread():
    global db_records
    while True:
        fresh = load_embeddings_from_db(DB_PATH)
        with db_records_lock:
            db_records = fresh
        time.sleep(EMBED_RELOAD_SEC)

# ==========================================
# DETECTION THREAD
# ==========================================

def detection_thread():
    while True:
        try:
            frame = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        faces = app.get(frame)   # includes .embedding when recognition module loaded

        try:
            if faces_queue.full():
                faces_queue.get_nowait()
            faces_queue.put_nowait(faces)
        except queue.Full:
            pass

# ==========================================
# TRACKING + RECOGNITION THREAD
# ==========================================

def tracking_thread():
    global draw_results, active_tracks, next_track_id

    batch_count = 0

    while True:
        try:
            faces = faces_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        current_time   = time.time()
        batch_count   += 1
        do_recognition = (batch_count % RECOG_EVERY_N == 0)

        with db_records_lock:
            current_db = list(db_records)

        matched_track_ids = set()

        for face in faces:
            new_box    = face.bbox.astype(int)
            new_center = (
                (new_box[0] + new_box[2]) / 2,
                (new_box[1] + new_box[3]) / 2,
            )

            # ---- find closest existing track ----
            best_track_id = -1
            min_dist      = MAX_MATCH_DIST

            with track_lock:
                for track_id, data in active_tracks.items():
                    if track_id in matched_track_ids:
                        continue
                    old_box    = data["box"]
                    old_center = (
                        (old_box[0] + old_box[2]) / 2,
                        (old_box[1] + old_box[3]) / 2,
                    )
                    dist = np.hypot(
                        new_center[0] - old_center[0],
                        new_center[1] - old_center[1],
                    )
                    if dist < min_dist:
                        min_dist      = dist
                        best_track_id = track_id

            # ---- smooth or create track ----
            if best_track_id != -1:
                with track_lock:
                    old_box   = active_tracks[best_track_id]["box"]
                final_box = [
                    int(old_box[0] * (1 - SMOOTH_ALPHA) + new_box[0] * SMOOTH_ALPHA),
                    int(old_box[1] * (1 - SMOOTH_ALPHA) + new_box[1] * SMOOTH_ALPHA),
                    int(old_box[2] * (1 - SMOOTH_ALPHA) + new_box[2] * SMOOTH_ALPHA),
                    int(old_box[3] * (1 - SMOOTH_ALPHA) + new_box[3] * SMOOTH_ALPHA),
                ]
                matched_track_ids.add(best_track_id)
                tid = best_track_id
            else:
                final_box = new_box.tolist()
                with track_lock:
                    tid           = next_track_id
                    next_track_id += 1
                matched_track_ids.add(tid)

            # ---- recognition (only every Nth batch, or new track) ----
            new_track = (best_track_id == -1)

            if (do_recognition or new_track) and hasattr(face, "embedding") and face.embedding is not None:
                name, role, sim = match_embedding(face.embedding, current_db)
            else:
                with track_lock:
                    prev = active_tracks.get(tid, {})
                name = prev.get("name", "Unknown")
                role = prev.get("role", "")
                sim  = prev.get("sim",  0.0)

            with track_lock:
                active_tracks[tid] = {
                    "box":       final_box,
                    "last_seen": current_time,
                    "name":      name,
                    "role":      role,
                    "sim":       sim,
                }

        # ---- expire stale tracks ----
        with track_lock:
            stale = [
                tid for tid, d in active_tracks.items()
                if current_time - d["last_seen"] > TRACK_TIMEOUT
            ]
            for tid in stale:
                del active_tracks[tid]

            temp = [
                {
                    "box":  d["box"],
                    "name": d["name"],
                    "role": d["role"],
                    "sim":  d["sim"],
                }
                for d in active_tracks.values()
            ]

        with results_lock:
            draw_results = temp

# ==========================================
# DRAWING HELPER
# ==========================================

def draw_face(frame, box, name, role, sim):
    x1, y1, x2, y2 = box
    known = (name != "Unknown")

    # Box colour: green for known, red for unknown
    colour = (0, 220, 0) if known else (0, 0, 220)
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

    # Label background
    label      = f"{name}" if known else "Unknown"
    sub_label  = f"{role}  {sim*100:.0f}%" if known else f"{sim*100:.0f}%"

    font       = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.55
    thickness  = 1

    (lw, lh), _ = cv2.getTextSize(label, font, font_scale, thickness)
    pad = 4

    # Main label box
    cv2.rectangle(
        frame,
        (x1, y1 - lh - pad * 2),
        (x1 + lw + pad * 2, y1),
        colour,
        cv2.FILLED,
    )
    cv2.putText(
        frame, label,
        (x1 + pad, y1 - pad),
        font, font_scale,
        (255, 255, 255),
        thickness, cv2.LINE_AA,
    )

    # Sub-label (role + confidence) below main box
    cv2.putText(
        frame, sub_label,
        (x1 + pad, y2 + lh + pad),
        font, 0.45,
        colour,
        1, cv2.LINE_AA,
    )

# ==========================================
# START THREADS
# ==========================================

threading.Thread(target=db_reload_thread,  daemon=True).start()
threading.Thread(target=detection_thread,  daemon=True).start()
threading.Thread(target=tracking_thread,   daemon=True).start()

# Give the DB loader a moment to populate before the loop starts
time.sleep(0.5)

# ==========================================
# MAIN LOOP
# ==========================================

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = resize_keep_aspect(frame)

    # Push latest frame to detection thread
    try:
        if frame_queue.full():
            frame_queue.get_nowait()
        frame_queue.put_nowait(frame.copy())
    except queue.Full:
        pass

    display = frame.copy()

    with results_lock:
        snapshot = list(draw_results)

    for res in snapshot:
        draw_face(display, res["box"], res["name"], res["role"], res["sim"])

    # HUD: staff count
    known_count = sum(1 for r in snapshot if r["name"] != "Unknown")
    cv2.putText(
        display,
        f"Faces: {len(snapshot)}  Known: {known_count}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (200, 200, 200), 1, cv2.LINE_AA,
    )

    cv2.imshow("Face Recognition", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()