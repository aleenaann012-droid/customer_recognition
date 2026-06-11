import numpy as np
class FaceCentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=150):
        self.next_id = 1
        self.objects = {} # id -> (cx, cy)
        self.disappeared = {} # id -> frames
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, faces):
        tracked_ids = []
        if len(faces) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    del self.objects[obj_id]
                    del self.disappeared[obj_id]
            return tracked_ids

        input_centroids = []
        for face in faces:
            cx = (face.bbox[0] + face.bbox[2]) / 2
            cy = (face.bbox[1] + face.bbox[3]) / 2
            input_centroids.append((cx, cy))

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.objects[self.next_id] = input_centroids[i]
                self.disappeared[self.next_id] = 0
                tracked_ids.append(self.next_id)
                self.next_id += 1
            return tracked_ids

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        D = np.zeros((len(object_ids), len(input_centroids)))
        for i, oc in enumerate(object_centroids):
            for j, ic in enumerate(input_centroids):
                D[i, j] = np.hypot(oc[0] - ic[0], oc[1] - ic[1])

        used_rows = set()
        used_cols = set()
        assignments = {}

        while len(used_rows) < len(object_ids) and len(used_cols) < len(input_centroids):
            min_val = np.min(D)
            if min_val > self.max_distance:
                break
            
            row, col = np.unravel_index(np.argmin(D), D.shape)
            
            obj_id = object_ids[row]
            assignments[col] = obj_id
            
            self.objects[obj_id] = input_centroids[col]
            self.disappeared[obj_id] = 0
            
            used_rows.add(row)
            used_cols.add(col)
            
            D[row, :] = 99999
            D[:, col] = 99999

        for row, obj_id in enumerate(object_ids):
            if row not in used_rows:
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    del self.objects[obj_id]
                    del self.disappeared[obj_id]

        for col in range(len(input_centroids)):
            if col not in used_cols:
                obj_id = self.next_id
                self.next_id += 1
                self.objects[obj_id] = input_centroids[col]
                self.disappeared[obj_id] = 0
                assignments[col] = obj_id
        
        for col in range(len(input_centroids)):
            tracked_ids.append(assignments[col])
            
        return tracked_ids

tracker = FaceCentroidTracker(
    max_disappeared=20,
    max_distance=150
)

class FaceByteTracker:
    def __init__(self, track_activation_threshold=0.3, lost_track_buffer=30, minimum_matching_threshold=0.9, minimum_consecutive_frames=1, max_distance=150):
        import supervision as sv
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning, module="supervision")
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            minimum_consecutive_frames=minimum_consecutive_frames
        )
        self.max_distance = max_distance

    def update(self, faces, detections):
        tracks = self.tracker.update_with_detections(detections)
        tracked_ids = []
        
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
                    if dist < self.max_distance and dist < best_dist:
                        best_dist = dist
                        matched_tid = int(t_id)
                
                # Fallback
                if matched_tid == i_face:
                    closest_tid = -1
                    min_d = 9999
                    for t_box, t_id in zip(tracks.xyxy, tracks.tracker_id):
                        tx_c = (t_box[0] + t_box[2]) / 2
                        ty_c = (t_box[1] + t_box[3]) / 2
                        dist = ((fx_c - tx_c)**2 + (fy_c - ty_c)**2)**0.5
                        if dist < min_d:
                            min_d = dist
                            closest_tid = int(t_id)
                    if closest_tid != -1:
                        matched_tid = closest_tid
            
            tracked_ids.append(matched_tid)
            
        return tracked_ids
