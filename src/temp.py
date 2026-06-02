import argparse
import cv2
import json
import numpy as np
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from collections import deque

# ─── Config ───────────────────────────────────────────────────────────────────

EMBEDDINGS_FILE   = "face_embeddings.json"
MATCH_THRESHOLD   = 0.45      # cosine distance — lower = stricter
RECOG_EVERY       = 8         # run InsightFace every N frames
LABEL_HOLD_FRAMES = 30        # keep last identity label for this many frames
WINDOW_NAME       = "FaceID"
FONT              = cv2.FONT_HERSHEY_SIMPLEX

# ─── InsightFace loader ───────────────────────────────────────────────────────

_iface_lock = threading.Lock()
_iface_app  = None

def get_insightface():
    global _iface_app
    if _iface_app is None:
        with _iface_lock:
            if _iface_app is None:
                try:
                    from insightface.app import FaceAnalysis
                except ImportError:
                    sys.exit(
                        "[ERROR] insightface not installed.\n"
                        "Run: pip install insightface onnxruntime opencv-python numpy mediapipe"
                    )
                print("[INFO] Loading InsightFace buffalo_s …")
                app = FaceAnalysis(
                    name="buffalo_s",
                    providers=["CPUExecutionProvider"],
                )
                app.prepare(ctx_id=0, det_size=(640, 640))   # 640 needed for multiple/distant faces
                _iface_app = app
                print("[INFO] InsightFace ready.")
    return _iface_app

# ─── MediaPipe loader ─────────────────────────────────────────────────────────

_mp_lock     = threading.Lock()
_mp_detector = None

def get_mediapipe():
    global _mp_detector
    if _mp_detector is None:
        with _mp_lock:
            if _mp_detector is None:
                try:
                    import mediapipe as mp
                except ImportError:
                    sys.exit("[ERROR] mediapipe not installed. Run: pip install mediapipe")
                # Support both new (>=0.10) and old (<0.10) MediaPipe APIs
                try:
                    from mediapipe.tasks.python import vision as mp_vision
                    from mediapipe.tasks.python import BaseOptions
                    import urllib.request
                    model_path = "blaze_face_short_range.tflite"
                    if not os.path.exists(model_path):
                        print("[INFO] Downloading MediaPipe face model (~1 MB)...")
                        urllib.request.urlretrieve(
                            "https://storage.googleapis.com/mediapipe-models/"
                            "face_detector/blaze_face_short_range/float16/1/"
                            "blaze_face_short_range.tflite",
                            model_path,
                        )
                    opts = mp_vision.FaceDetectorOptions(
                        base_options=BaseOptions(model_asset_path=model_path),
                        min_detection_confidence=0.5,
                    )
                    det = mp_vision.FaceDetector.create_from_options(opts)
                    det._new_api = True
                    print("[INFO] MediaPipe FaceDetector ready (new API >=0.10).")
                except Exception:
                    det = mp.solutions.face_detection.FaceDetection(
                        model_selection=1,
                        min_detection_confidence=0.5,
                    )
                    det._new_api = False
                    print("[INFO] MediaPipe FaceDetection ready (legacy API <0.10).")
                _mp_detector = det
    return _mp_detector


# ─── Embedding helpers ────────────────────────────────────────────────────────

def normalise(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(normalise(a), normalise(b)))

# ─── JSON database ────────────────────────────────────────────────────────────

def load_db(path: str = EMBEDDINGS_FILE) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = json.load(f)
    return {name: np.array(emb, dtype=np.float32) for name, emb in raw.items()}

def save_db(db: dict, path: str = EMBEDDINGS_FILE):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({n: e.tolist() for n, e in db.items()}, f, indent=2)
    os.replace(tmp, path)
    print(f"[INFO] Saved {len(db)} face(s) → {path}")

# ─── Threaded matching ────────────────────────────────────────────────────────

def match_embedding(query_emb: np.ndarray, db: dict, threshold: float = MATCH_THRESHOLD):
    if not db:
        return None, 1.0
    best_name, best_dist = None, threshold
    with ThreadPoolExecutor(max_workers=min(8, len(db))) as pool:
        futs = {pool.submit(cosine_distance, emb, query_emb): name for name, emb in db.items()}
        for fut, name in futs.items():
            dist = fut.result()
            if dist < best_dist:
                best_dist, best_name = dist, name
    return best_name, best_dist

# ─── MediaPipe detection (every frame) ───────────────────────────────────────

def mp_detect(frame_bgr: np.ndarray) -> list:
    """
    Run MediaPipe on a BGR frame.
    Returns list of [x1, y1, x2, y2] bounding boxes.
    Handles both new (>=0.10) and old (<0.10) MediaPipe APIs.
    """
    det  = get_mediapipe()
    h, w = frame_bgr.shape[:2]
    boxes = []
    if getattr(det, "_new_api", False):
        import mediapipe as mp
        rgb      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = det.detect(mp_image)
        for d in result.detections:
            bb = d.bounding_box
            x1 = max(0, bb.origin_x)
            y1 = max(0, bb.origin_y)
            x2 = min(w, bb.origin_x + bb.width)
            y2 = min(h, bb.origin_y + bb.height)
            boxes.append([x1, y1, x2, y2])
    else:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = det.process(rgb)
        if res.detections:
            for d in res.detections:
                bb = d.location_data.relative_bounding_box
                x1 = max(0, int(bb.xmin * w))
                y1 = max(0, int(bb.ymin * h))
                x2 = min(w, int((bb.xmin + bb.width)  * w))
                y2 = min(h, int((bb.ymin + bb.height) * h))
                boxes.append([x1, y1, x2, y2])
    return boxes


# ─── InsightFace embedding (every N frames) ───────────────────────────────────

def iface_embed(frame_bgr: np.ndarray) -> list:
    """
    Run InsightFace on a BGR frame.
    Returns list of (bbox_array, embedding).
    """
    app   = get_insightface()
    faces = app.get(frame_bgr)
    return [(f.bbox.astype(int), f.normed_embedding) for f in faces]

# ─── IOU helper ──────────────────────────────────────────────────────────────

def iou(a, b) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

# ─── Drawing ──────────────────────────────────────────────────────────────────

def draw_box(frame, bbox, label: str, colour=(0, 230, 0), alpha: float = 1.0):
    """Draw a rounded-corner box with a filled label badge."""
    x1, y1, x2, y2 = [int(v) for v in bbox]

    # Fade colour by alpha (for label hold-off)
    c = tuple(int(ch * alpha) for ch in colour)

    # Draw rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)

    # Corner accents
    ln = 18
    cv2.line(frame, (x1, y1), (x1+ln, y1), c, 3)
    cv2.line(frame, (x1, y1), (x1, y1+ln), c, 3)
    cv2.line(frame, (x2, y1), (x2-ln, y1), c, 3)
    cv2.line(frame, (x2, y1), (x2, y1+ln), c, 3)
    cv2.line(frame, (x1, y2), (x1+ln, y2), c, 3)
    cv2.line(frame, (x1, y2), (x1, y2-ln), c, 3)
    cv2.line(frame, (x2, y2), (x2-ln, y2), c, 3)
    cv2.line(frame, (x2, y2), (x2, y2-ln), c, 3)

    if label:
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 2)
        cy = y1 - 6 if y1 > th + 10 else y2 + th + 10
        cv2.rectangle(frame, (x1, cy-th-4), (x1+tw+6, cy+4), c, -1)
        cv2.putText(frame, label, (x1+3, cy), FONT, 0.6, (0,0,0), 2)

# ─── Identity cache (maps bbox → identity label) ──────────────────────────────

class IdentityCache:
    """
    Stores the last known identity for each face, keyed by bbox centroid.
    Allows the display to keep showing a name even when InsightFace
    isn't running on this frame.
    """
    def __init__(self, hold: int = LABEL_HOLD_FRAMES):
        self.hold    = hold
        self.entries = []   # list of {cx, cy, label, colour, ttl}

    def update_from_recog(self, mp_boxes: list, recog_results: list):
        """
        Match recognition results (InsightFace bbox + label) to
        current MediaPipe boxes and refresh cache entries.
        recog_results: list of (bbox, label, colour)
        """
        for rbbox, label, colour in recog_results:
            rcx = (rbbox[0]+rbbox[2])//2
            rcy = (rbbox[1]+rbbox[3])//2
            # find closest MediaPipe box to this recognition result
            best_idx, best_dist = -1, 9999
            for i, mb in enumerate(mp_boxes):
                mcx = (mb[0]+mb[2])//2
                mcy = (mb[1]+mb[3])//2
                d   = abs(rcx-mcx) + abs(rcy-mcy)
                if d < best_dist:
                    best_dist, best_idx = d, i
            # update or create cache entry
            if best_idx >= 0:
                mb  = mp_boxes[best_idx]
                cx  = (mb[0]+mb[2])//2
                cy  = (mb[1]+mb[3])//2
                # find existing entry
                matched = False
                for e in self.entries:
                    if abs(e["cx"]-cx) < 80 and abs(e["cy"]-cy) < 80:
                        e["label"]  = label
                        e["colour"] = colour
                        e["ttl"]    = self.hold
                        e["cx"]     = cx
                        e["cy"]     = cy
                        matched = True
                        break
                if not matched:
                    self.entries.append({"cx": cx, "cy": cy,
                                         "label": label, "colour": colour,
                                         "ttl": self.hold})

    def get_label_for_box(self, bbox):
        """Return (label, colour, alpha) for the given MediaPipe bbox, or None."""
        cx = (bbox[0]+bbox[2])//2
        cy = (bbox[1]+bbox[3])//2
        best = None
        best_dist = 120   # max centroid distance (px) to consider a match
        for e in self.entries:
            d = ((e["cx"]-cx)**2 + (e["cy"]-cy)**2) ** 0.5
            if d < best_dist:
                best_dist = d
                best = e
        if best:
            alpha = min(1.0, best["ttl"] / (self.hold * 0.3))
            return best["label"], best["colour"], alpha
        return None, (180,180,180), 0.5

    def tick(self):
        """Call once per display frame to age out old entries."""
        self.entries = [e for e in self.entries if e["ttl"] > 0]
        for e in self.entries:
            e["ttl"] -= 1

# ─── REGISTER mode ────────────────────────────────────────────────────────────

def register_face(name: str, db_path: str = EMBEDDINGS_FILE):
    db  = load_db(db_path)
    camera_id = "PRIVATE_URL"
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open camera {camera_id}")

    print(f"\n[REGISTER] Name: '{name}'")
    print("  → Look at the camera, then press  SPACE  to capture.")
    print("  → Press  Q  to quit.\n")

    # Pre-load models
    get_mediapipe()
    get_insightface()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display   = frame.copy()
        mp_boxes  = mp_detect(frame)

        for bbox in mp_boxes:
            draw_box(display, bbox, "Detected — SPACE to capture", (0, 200, 255))

        cv2.putText(display, f"Registering: {name}", (10, 30),
                    FONT, 0.75, (0, 230, 255), 2)
        cv2.putText(display, f"Faces in frame: {len(mp_boxes)}", (10, 58),
                    FONT, 0.6, (255,255,255), 1)
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] Registration cancelled.")
            break
        elif key == ord(" "):
            if len(mp_boxes) == 0:
                print("[WARN] No face detected — try again.")
            elif len(mp_boxes) > 1:
                print("[WARN] Multiple faces — only one person should be in frame.")
            else:
                faces = iface_embed(frame)
                if not faces:
                    print("[WARN] InsightFace couldn't extract embedding — try again.")
                    continue
                _, emb = faces[0]
                if name in db:
                    print(f"[WARN] '{name}' already exists — overwriting.")
                db[name] = emb
                save_db(db, db_path)
                print(f"[OK]  Registered '{name}'!")

                confirm = frame.copy()
                cv2.putText(confirm, f"Registered: {name}", (30, 60),
                            FONT, 1.2, (0,255,0), 3)
                cv2.imshow(WINDOW_NAME, confirm)
                cv2.waitKey(1500)
                break

    cap.release()
    cv2.destroyAllWindows()

# ─── MATCH mode ───────────────────────────────────────────────────────────────

class RecogState:
    """Shared state between display thread and recognition thread."""
    def __init__(self):
        self.lock          = threading.Lock()
        self.latest_frame  = None          # frame sent to recog thread
        self.recog_results = []            # [(bbox, label, colour), ...]
        self.mp_boxes_snap = []            # MP boxes at time of recog frame
        self.stop          = False
        self.recog_fps     = 0.0

def _recog_worker(state: RecogState, db: dict, threshold: float):
    """
    Background thread: run InsightFace + matching on frames
    delivered by the main loop.
    """
    times = deque(maxlen=20)

    while not state.stop:
        with state.lock:
            frame    = state.latest_frame
            mp_boxes = state.mp_boxes_snap

        if frame is None:
            time.sleep(0.005)
            continue

        t0    = time.perf_counter()
        faces = iface_embed(frame)
        results = []

        with ThreadPoolExecutor(max_workers=min(8, max(1, len(db)))) as pool:
            futs = {pool.submit(match_embedding, emb, db, threshold): bbox
                    for bbox, emb in faces}
            for fut, bbox in futs.items():
                name, dist = fut.result()
                if name:
                    label  = f"{name}  {(1-dist)*100:.1f}%"
                    colour = (0, 230, 0)
                else:
                    label  = f"Unknown  {(1-dist)*100:.1f}%"
                    colour = (0, 0, 220)
                results.append((bbox, label, colour))

        times.append(time.perf_counter() - t0)

        with state.lock:
            state.recog_results = results
            state.latest_frame  = None
            state.recog_fps     = 1.0 / (sum(times)/len(times)) if times else 0.0


def match_faces(
    db_path:   str   = EMBEDDINGS_FILE,
    threshold: float = MATCH_THRESHOLD
):
    db = load_db(db_path)
    if not db:
        print(f"[WARN] No embeddings in '{db_path}'. Register faces first.")

    print(f"\n[MATCH] {len(db)} registered face(s): {', '.join(db.keys()) or 'none'}")
    print("  → Press  Q  to quit.\n")

    # Pre-load both models before opening camera
    get_mediapipe()
    get_insightface()

    camera_id = "PRIVATE_URL"
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open camera {camera_id}")

    state  = RecogState()
    cache  = IdentityCache(hold=LABEL_HOLD_FRAMES)

    worker = threading.Thread(
        target=_recog_worker,
        args=(state, db, threshold),
        daemon=True,
    )
    worker.start()

    frame_count  = 0
    display_times = deque(maxlen=30)

    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # ── 1. MediaPipe: detect on EVERY frame (fast, smooth) ──
        mp_boxes = mp_detect(frame)

        # ── 2. Every N frames: ship frame to InsightFace thread ──
        if frame_count % RECOG_EVERY == 0:
            with state.lock:
                if state.latest_frame is None:      # don't queue if busy
                    state.latest_frame  = frame.copy()
                    state.mp_boxes_snap = mp_boxes[:]

        # ── 3. Merge fresh recognition results into cache ──
        with state.lock:
            fresh   = state.recog_results
            rfps    = state.recog_fps
            state.recog_results = []               # consume

        if fresh:
            cache.update_from_recog(mp_boxes, fresh)

        cache.tick()

        # ── 4. Draw: MediaPipe box + cached label ──
        display = frame.copy()
        for bbox in mp_boxes:
            label, colour, alpha = cache.get_label_for_box(bbox)
            draw_box(display, bbox, label or "Detecting…", colour, alpha)

        # ── 5. HUD ──
        display_times.append(time.perf_counter() - t0)
        dfps = 1.0 / (sum(display_times)/len(display_times)) if display_times else 0
        ts   = datetime.now().strftime("%H:%M:%S")
        cv2.putText(display,
                    f"Display: {dfps:.0f} fps  |  Recog: {rfps:.1f} fps  |  {ts}",
                    (10, 28), FONT, 0.6, (255,255,255), 2)
        cv2.putText(display,
                    f"DB: {len(db)} face(s)   Q = quit",
                    (10, 52), FONT, 0.5, (200,200,200), 1)
        cv2.imshow(WINDOW_NAME, display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    state.stop = True
    worker.join(timeout=3)
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Session ended.")

# ─── List / Delete ────────────────────────────────────────────────────────────

def list_faces(db_path: str = EMBEDDINGS_FILE):
    db = load_db(db_path)
    if not db:
        print("No registered faces.")
        return
    print(f"\nRegistered faces ({db_path}):")
    for i, n in enumerate(sorted(db), 1):
        print(f"  {i:>3}. {n}")
    print(f"\nTotal: {len(db)}")

def delete_face(name: str, db_path: str = EMBEDDINGS_FILE):
    db = load_db(db_path)
    if name not in db:
        print(f"[WARN] '{name}' not found.")
        return
    del db[name]
    save_db(db, db_path)
    print(f"[OK] Deleted '{name}'.")

# ─── CLI ──────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="FaceID — MediaPipe tracking + InsightFace recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python face_recognition.py register --name "Alice"
  python face_recognition.py match
  python face_recognition.py match --threshold 0.50 --camera 1
  python face_recognition.py list
  python face_recognition.py delete --name "Alice"
        """,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    reg = sub.add_parser("register")
    reg.add_argument("--name",   required=True)
    reg.add_argument("--db",     default=EMBEDDINGS_FILE)
    reg.add_argument("--camera", type=int, default=0)

    mat = sub.add_parser("match")
    mat.add_argument("--db",        default=EMBEDDINGS_FILE)
    mat.add_argument("--threshold", type=float, default=MATCH_THRESHOLD)
   # mat.add_argument("--camera",    type=int, default=0)

    lst = sub.add_parser("list")
    lst.add_argument("--db", default=EMBEDDINGS_FILE)

    dlt = sub.add_parser("delete")
    dlt.add_argument("--name", required=True)
    dlt.add_argument("--db",   default=EMBEDDINGS_FILE)

    return p

def main():
    args = build_parser().parse_args()
    if args.mode == "register":
        register_face(args.name, db_path=args.db)  #  ,camera_id=args.camera
    elif args.mode == "match":
        match_faces(db_path=args.db, threshold=args.threshold) # , camera_id=args.camera
    elif args.mode == "list":
        list_faces(args.db)
    elif args.mode == "delete":
        delete_face(args.name, db_path=args.db)

if __name__ == "__main__":
    main()
