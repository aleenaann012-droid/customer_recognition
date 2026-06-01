import argparse
import cv2
import json
import numpy as np
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime


EMBEDDINGS_FILE = "face_embeddings.json"
MATCH_THRESHOLD  = 0.45          
FRAME_SKIP       = 2            
WINDOW_NAME      = "FaceID"
FONT             = cv2.FONT_HERSHEY_SIMPLEX


_app_lock = threading.Lock()
_app      = None

def get_app():
    """Return a cached InsightFace FaceAnalysis app (thread-safe)."""
    global _app
    if _app is None:
        with _app_lock:
            if _app is None:
                try:
                    import insightface
                    from insightface.app import FaceAnalysis
                except ImportError:
                    sys.exit(
                        "[ERROR] insightface not installed.\n"
                        "Run: pip install insightface onnxruntime opencv-python numpy"
                    )
                print("[INFO] Loading InsightFace model (first run downloads ~500 MB)…")

                # Auto-select best available provider (TensorRT skipped - requires extra DLLs)
                import onnxruntime as ort
                available = ort.get_available_providers()
                if "CUDAExecutionProvider" in available:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    print("[INFO] Backend: CUDA ⚡")
                else:
                    providers = ["CPUExecutionProvider"]
                    print("[WARN] Backend: CPU only (no GPU detected)")

                app = FaceAnalysis(
                    name="buffalo_s",          # small/fast model – lower latency, smoother tracking
                    providers=providers,
                )
                app.prepare(ctx_id=0, det_size=(640, 640))
                _app = app
                print("[INFO] Model ready.")
    return _app



def normalise(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(normalise(a), normalise(b)))



def load_db(path: str = EMBEDDINGS_FILE) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        raw = json.load(f)
    return {name: np.array(emb, dtype=np.float32) for name, emb in raw.items()}


def save_db(db: dict, path: str = EMBEDDINGS_FILE) -> None:
    serialisable = {name: emb.tolist() for name, emb in db.items()}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(serialisable, f, indent=2)
    os.replace(tmp, path)          # atomic write
    print(f"[INFO] Database saved → {path}  ({len(db)} entries)")



def _compare_one(name: str, stored_emb: np.ndarray, query_emb: np.ndarray):
    """Worker: compute cosine distance for one entry."""
    dist = cosine_distance(stored_emb, query_emb)
    return name, dist


def match_embedding(query_emb: np.ndarray, db: dict, threshold: float = MATCH_THRESHOLD):
    """
    Compare query_emb against every entry in db using a thread pool.
    Returns (best_name, best_distance) or (None, 1.0) if no match.
    """
    if not db:
        return None, 1.0

    best_name = None
    best_dist = threshold         

    with ThreadPoolExecutor(max_workers=min(8, len(db))) as pool:
        futures = {
            pool.submit(_compare_one, name, emb, query_emb): name
            for name, emb in db.items()
        }
        for future in as_completed(futures):
            name, dist = future.result()
            if dist < best_dist:
                best_dist = dist
                best_name = name

    return best_name, best_dist




def detect_and_embed(frame_bgr: np.ndarray):
    """Run InsightFace on a BGR frame; return list of (bbox, embedding, kps)."""
    app   = get_app()
    faces = app.get(frame_bgr)
    results = []
    for face in faces:
        bbox = face.bbox.astype(int)           
        emb  = face.normed_embedding            
        kps  = getattr(face, "kps", None)
        results.append((bbox, emb, kps))
    return results


def draw_box(frame, bbox, label: str, colour=(0, 230, 0)):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.65, 2)
    cy = y1 - 8 if y1 > th + 8 else y2 + th + 8
    cv2.rectangle(frame, (x1, cy - th - 4), (x1 + tw + 4, cy + 4), colour, -1)
    cv2.putText(frame, label, (x1 + 2, cy), FONT, 0.65, (0, 0, 0), 2)


class BBoxKalman:
    """
    4-state Kalman filter tracking [x1, y1, x2, y2] with velocity.
    Produces smooth, interpolated boxes between detections.
    """
    def __init__(self, bbox):
        self.kf = cv2.KalmanFilter(8, 4)   # state: x1 y1 x2 y2 dx dy dw dh
        dt = 1.0
        self.kf.transitionMatrix = np.array([
            [1,0,0,0, dt,0,0,0],
            [0,1,0,0, 0,dt,0,0],
            [0,0,1,0, 0,0,dt,0],
            [0,0,0,1, 0,0,0,dt],
            [0,0,0,0, 1,0,0,0],
            [0,0,0,0, 0,1,0,0],
            [0,0,0,0, 0,0,1,0],
            [0,0,0,0, 0,0,0,1],
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)
        self.kf.processNoiseCov    = np.eye(8, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov= np.eye(4, dtype=np.float32) * 0.5
        self.kf.errorCovPost       = np.eye(8, dtype=np.float32)
        self.kf.statePost = np.array(
            [bbox[0], bbox[1], bbox[2], bbox[3], 0, 0, 0, 0],
            dtype=np.float32
        ).reshape(8, 1)
        self.missed = 0

    def predict(self) -> np.ndarray:
        pred = self.kf.predict()
        return pred[:4].flatten().astype(int)

    def update(self, bbox):
        meas = np.array(bbox, dtype=np.float32).reshape(4, 1)
        self.kf.correct(meas)
        self.missed = 0

    def get(self) -> np.ndarray:
        return self.kf.statePost[:4].flatten().astype(int)


def iou(a, b) -> float:
    """Intersection-over-Union of two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter)


class FaceTracker:

    MAX_MISS  = 8     # frames before a track is discarded
    IOU_THRESH = 0.25 # minimum IOU to consider a detection a match

    def __init__(self):
        self.tracks: dict[int, dict] = {}   # id → {kalman, label, colour, missed}
        self._next_id = 0

    def _new_id(self):
        tid = self._next_id
        self._next_id += 1
        return tid

    def update(self, detections: list) -> list:
        """
        detections: list of (bbox, label, colour)
        Returns:    list of (smoothed_bbox, label, colour)
        """
        # --- predict step for all existing tracks ---
        for t in self.tracks.values():
            t["predicted"] = t["kalman"].predict()

        # --- greedy IOU matching ---
        matched_track_ids  = set()
        matched_det_ids    = set()

        det_bboxes = [d[0] for d in detections]

        for tid, t in self.tracks.items():
            best_iou = self.IOU_THRESH
            best_did = -1
            for did, dbbox in enumerate(det_bboxes):
                if did in matched_det_ids:
                    continue
                score = iou(t["predicted"], dbbox)
                if score > best_iou:
                    best_iou = score
                    best_did = did
            if best_did >= 0:
                dbbox, dlabel, dcolour = detections[best_did]
                t["kalman"].update(dbbox)
                t["label"]  = dlabel
                t["colour"] = dcolour
                t["missed"] = 0
                matched_track_ids.add(tid)
                matched_det_ids.add(best_did)

        # --- increment miss counter for unmatched tracks ---
        for tid in list(self.tracks.keys()):
            if tid not in matched_track_ids:
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > self.MAX_MISS:
                    del self.tracks[tid]

        # --- spawn new tracks for unmatched detections ---
        for did, (dbbox, dlabel, dcolour) in enumerate(detections):
            if did not in matched_det_ids:
                tid = self._new_id()
                self.tracks[tid] = {
                    "kalman": BBoxKalman(dbbox),
                    "label":  dlabel,
                    "colour": dcolour,
                    "missed": 0,
                }

        # --- output smoothed results ---
        out = []
        for t in self.tracks.values():
            smoothed = t["kalman"].get()
            out.append((smoothed, t["label"], t["colour"]))
        return out



def register_face(name: str, db_path: str = EMBEDDINGS_FILE, camera_id: int = 0):
    """
    Open webcam, wait until exactly one face is detected, save embedding.
    Press SPACE to capture, Q to quit without saving.
    """
    db  = load_db(db_path)
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open camera {camera_id}")

    print(f"\n[REGISTER] Name: '{name}'")
    print("  → Look at the camera and press  SPACE  to capture.")
    print("  → Press  Q  to quit.\n")

    frame_count = 0
    detected    = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        frame_count += 1

        if frame_count % FRAME_SKIP == 0:
            detected = detect_and_embed(frame)

        for bbox, emb, _ in detected:
            draw_box(display, bbox, "Detected – SPACE to capture", (0, 200, 255))

        status = f"Faces: {len(detected)}"
        cv2.putText(display, status, (10, 28), FONT, 0.7, (255, 255, 255), 2)
        cv2.putText(display, f"Registering: {name}", (10, 56), FONT, 0.7, (0, 230, 255), 2)
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] Registration cancelled.")
            break
        elif key == ord(" "):
            if len(detected) == 0:
                print("[WARN] No face detected – please try again.")
            elif len(detected) > 1:
                print("[WARN] Multiple faces detected – only one person should be in frame.")
            else:
                _, emb, _ = detected[0]
                if name in db:
                    print(f"[WARN] '{name}' already exists – overwriting.")
                db[name] = emb
                save_db(db, db_path)
                print(f"[OK]   Registered '{name}' successfully!")

                # Show confirmation flash
                confirm = frame.copy()
                cv2.putText(confirm, f"Registered: {name}", (30, 60),
                            FONT, 1.2, (0, 255, 0), 3)
                cv2.imshow(WINDOW_NAME, confirm)
                cv2.waitKey(1200)
                break

    cap.release()
    cv2.destroyAllWindows()



class MatchState:
    """Shared state between capture and match threads."""
    def __init__(self):
        self.lock        = threading.Lock()
        self.latest_frame: np.ndarray | None = None
        self.raw_detections: list = []   # raw (bbox, label, colour) from worker
        self.results: list = []          # smoothed (bbox, label, colour) for display
        self.stop        = False
        self.fps_display = 0.0


def _match_worker(state: MatchState, db: dict, threshold: float):
    """
    Background thread: pull the latest frame, run detection + matching,
    write annotated results back to state.
    """
    app = get_app()
    frame_times = []

    while not state.stop:
        with state.lock:
            frame = state.latest_frame

        if frame is None:
            time.sleep(0.005)
            continue

        t0      = time.perf_counter()
        faces   = detect_and_embed(frame)
        results = []

        match_futures = {}
        with ThreadPoolExecutor(max_workers=min(8, max(1, len(db)))) as pool:
            for bbox, emb, kps in faces:
                fut = pool.submit(match_embedding, emb, db, threshold)
                match_futures[fut] = bbox

            for fut, bbox in match_futures.items():
                matched_name, dist = fut.result()
                if matched_name:
                    label  = f"{matched_name}  {(1-dist)*100:.1f}%"
                    colour = (0, 230, 0)
                else:
                    label  = f"Unknown  {(1-dist)*100:.1f}%"
                    colour = (0, 0, 230)
                results.append((bbox, label, colour))

        t1 = time.perf_counter()
        frame_times.append(t1 - t0)
        if len(frame_times) > 20:
            frame_times.pop(0)

        with state.lock:
            state.raw_detections = results
            state.fps_display    = 1.0 / (sum(frame_times) / len(frame_times))
            state.latest_frame   = None    # mark consumed


def match_faces(
    db_path: str  = EMBEDDINGS_FILE,
    threshold: float = MATCH_THRESHOLD,
    camera_id: int   = 0,
):
    """Open webcam and continuously match detected faces against the database."""
    db = load_db(db_path)
    if not db:
        print(f"[WARN] No embeddings found in '{db_path}'. Register faces first.")

    print(f"\n[MATCH] Loaded {len(db)} registered face(s): {', '.join(db.keys()) or 'none'}")
    print("  → Press  Q  to quit.\n")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open camera {camera_id}")

    state   = MatchState()
    tracker = FaceTracker()          # <── Kalman smoother lives here

    worker = threading.Thread(
        target=_match_worker,
        args=(state, db, threshold),
        daemon=True,
    )
    worker.start()

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Feed a new frame to the worker every FRAME_SKIP frames
        if frame_count % FRAME_SKIP == 0:
            with state.lock:
                state.latest_frame = frame.copy()

        # Pull latest raw detections & run smoother on every display frame
        display = frame.copy()
        with state.lock:
            raw_dets    = state.raw_detections
            fps_display = state.fps_display

        smoothed = tracker.update(raw_dets)
        for bbox, label, colour in smoothed:
            draw_box(display, bbox, label, colour)

        ts = datetime.now().strftime("%H:%M:%S")
        cv2.putText(display, f"FPS: {fps_display:.1f}  |  {ts}",
                    (10, 28), FONT, 0.65, (255, 255, 255), 2)
        cv2.putText(display, f"DB: {len(db)} face(s)  |  Q = quit",
                    (10, 54), FONT, 0.55, (200, 200, 200), 1)
        cv2.imshow(WINDOW_NAME, display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    state.stop = True
    worker.join(timeout=3)
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Match session ended.")


# ─── List registered faces ────────────────────────────────────────────────────

def list_faces(db_path: str = EMBEDDINGS_FILE):
    db = load_db(db_path)
    if not db:
        print("No registered faces found.")
        return
    print(f"\nRegistered faces in '{db_path}':")
    for i, name in enumerate(sorted(db), 1):
        print(f"  {i:>3}. {name}")
    print(f"\nTotal: {len(db)}")


def delete_face(name: str, db_path: str = EMBEDDINGS_FILE):
    db = load_db(db_path)
    if name not in db:
        print(f"[WARN] '{name}' not found in database.")
        return
    del db[name]
    save_db(db, db_path)
    print(f"[OK] Deleted '{name}' from database.")



def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Face Recognition with InsightFace  |  register · match · list · delete",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python face_recognition.py register --name "Alice"
  python face_recognition.py match
  python face_recognition.py match --threshold 0.40 --camera 1
  python face_recognition.py list
  python face_recognition.py delete --name "Alice"
        """,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # register
    reg = sub.add_parser("register", help="Register a new face")
    reg.add_argument("--name",     required=True, help="Person's name")
    reg.add_argument("--db",       default=EMBEDDINGS_FILE, help="Path to embeddings JSON")
    reg.add_argument("--camera",   type=int, default=0, help="Camera device index")

    # match
    mat = sub.add_parser("match", help="Run real-time face matching")
    mat.add_argument("--db",       default=EMBEDDINGS_FILE, help="Path to embeddings JSON")
    mat.add_argument("--threshold",type=float, default=MATCH_THRESHOLD,
                     help=f"Cosine distance threshold (default {MATCH_THRESHOLD})")
    mat.add_argument("--camera",   type=int, default=0, help="Camera device index")

    # list
    lst = sub.add_parser("list", help="List registered faces")
    lst.add_argument("--db", default=EMBEDDINGS_FILE)

    # delete
    dlt = sub.add_parser("delete", help="Delete a registered face")
    dlt.add_argument("--name", required=True)
    dlt.add_argument("--db",   default=EMBEDDINGS_FILE)

    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.mode == "register":
        register_face(args.name, db_path=args.db, camera_id=args.camera)

    elif args.mode == "match":
        match_faces(db_path=args.db, threshold=args.threshold, camera_id=args.camera)

    elif args.mode == "list":
        list_faces(args.db)

    elif args.mode == "delete":
        delete_face(args.name, db_path=args.db)


if __name__ == "__main__":
    main()
