import queue
import numpy as np
import supervision as sv
import state
from models import app
def detection_thread():

    while True:
        try:
            frame = state.frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        state.frame_counter += 1

        faces = app.get(frame)
        
        valid_faces = []
        boxes=[]
        scores=[]
        for face in faces:
            bbox = face.bbox
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            
            # Filter out non-human false positives and tiny distant faces (under 25x25 pixels)
            if face.det_score >= 0.55 and w >= 25 and h >= 25:
                boxes.append(bbox)
                scores.append(face.det_score)
                valid_faces.append(face)
                
        faces = valid_faces
        if len(boxes) > 0:
            detections = sv.Detections(
                xyxy=np.array(boxes),
                confidence=np.array(scores),
                class_id=np.zeros(len(boxes), dtype=int)
            )
        else:
            detections = sv.Detections.empty()

        # Always push the latest result, drop stale one if needed
        try:
            if state.det_queue.full():
                state.det_queue.get_nowait()
            state.det_queue.put_nowait((faces, detections))
        except queue.Full:
            pass