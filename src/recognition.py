import queue
import numpy as np
import queue
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import state
from database import load_staff, insert_staff_attendance, insert_visitor, load_customers, insert_customer, update_customer, update_staff_exit, update_customer_exit
from tracker import FaceCentroidTracker, FaceByteTracker
import sys
import os
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from similarity import calculate_customer_similarity

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

staff_details, staff_embeddings = load_staff()
customer_details, customer_embeddings = load_customers()

if getattr(config, "TRACKER_TYPE", "centroid") == "bytetrack":
    logging.info("[INFO] Initializing ByteTrack Backend...")
    tracker = FaceByteTracker(
        lost_track_buffer=getattr(config, "TRACKER_MAX_DISAPPEARED", 30),
        max_distance=getattr(config, "TRACKER_MAX_DISTANCE", 150)
    )
else:
    print("[INFO] Initializing Centroid Tracker Backend...")
    tracker = FaceCentroidTracker(
        max_disappeared=getattr(config, "TRACKER_MAX_DISAPPEARED", 30),
        max_distance=getattr(config, "TRACKER_MAX_DISTANCE", 150)
    )

from utils import get_age_group
import time

track_history = {}

def recognition_thread():
     import collections
     while True:
        try:
            faces, detections = state.det_queue.get(timeout=1.0)
            if getattr(config, "TRACKER_TYPE", "centroid") == "bytetrack":
                tracked_ids = tracker.update(faces, detections)
            else:
                tracked_ids = tracker.update(faces)
        except queue.Empty:
            continue

        temp_results = []
        for i_face, face in enumerate(faces):
            matched_tid = tracked_ids[i_face] if i_face < len(tracked_ids) else i_face
            
            if matched_tid not in track_history:
                track_history[matched_tid] = {
                    'frames': 0,
                    'is_staff': False,
                    'processed': False,
                    'staff_name': None,
                    'staff_role': None,
                    'ages': [],
                    'genders': [],
                    'embeddings': []
                }
                
            th = track_history[matched_tid]
            th['frames'] += 1
            th['last_seen'] = time.time()
            
            embedding    = face.embedding.reshape(1, -1)
            age          = int(face.age)
            gender       = "Male" if face.gender == 1 else "Female"

            th['ages'].append(age)
            th['genders'].append(gender)
            th['embeddings'].append(face.embedding)

            for i, known_emb in enumerate(staff_embeddings):
                sim = cosine_similarity(
                        embedding,
                        known_emb.reshape(1, -1)
                    )[0][0]

                if sim > 0.3:
                    th['is_staff'] = True
                    th['staff_name'] = staff_details[i]["name"]
                    th['staff_role'] = staff_details[i]["role"]
                    th['staff_id'] = staff_details[i]["staff_id"]

                    staff_name = th['staff_name']
                    if staff_name not in state.staff_presence:
                        try:
                            logging.info(f"Triggering insert_staff_attendance for staff_name={staff_name}, staff_id={staff_details[i]['staff_id']}")
                            insert_staff_attendance(staff_details[i]["staff_id"])
                            state.staff_presence[staff_name] = True
                            logging.info(f"Staff attendance successfully processed for {staff_name}")
                        except Exception as e:
                            logging.error(f"Error inserting staff attendance for {staff_name}: {e}")
                    break

            if th['is_staff']:
                display_name = f"{th['staff_name']} | {th['staff_role']}"
            else:
                display_name = "CUSTOMER"

            text = f"ID:{matched_tid} | {display_name} | {gender} | Age:{age}"
            
            if not th['is_staff'] and not th['processed'] and th['frames'] >= 20:
                is_returning = False
                matched_cid = None
                
                final_gender = max(set(th['genders']), key=th['genders'].count)
                final_age = int(np.median(th['ages']))
                final_embedding = th['embeddings'][-1].reshape(1, -1)
                
                if len(customer_embeddings) > 0:
                    max_sim, best_idx = calculate_customer_similarity(final_embedding, customer_embeddings)
                    if max_sim > 0.6:
                        is_returning = True
                        matched_cid = customer_details[best_idx]["customer_id"]
                
                age_group = get_age_group(final_age)
                try:
                    if is_returning:
                        logging.info(f"Returning customer detected (ID: {matched_cid}). Triggering update_customer.")
                        update_customer(matched_cid)
                        th['customer_id'] = matched_cid
                    else:
                        logging.info(f"New customer detected. Triggering insert_visitor and insert_customer.")
                        insert_visitor(age_group, final_gender)
                        new_id = insert_customer(final_age, final_gender, th['embeddings'][-1])
                        
                        if new_id is not None:
                            customer_embeddings.append(th['embeddings'][-1])
                            customer_details.append({"customer_id": new_id})
                            th['customer_id'] = new_id
                            logging.info(f"New customer {new_id} successfully added to local cache.")
                        else:
                            logging.error("Failed to insert new customer; new_id is None.")
                        
                    th['processed'] = True
                except Exception as e:
                    logging.error(f"Error processing customer stats (insert/update): {e}")
            
            temp_results.append({
                "track_id": matched_tid,
                "box": face.bbox,
                "text": text
            })

        current_time = time.time()
        expired_tids = []
        for tid, t_hist in track_history.items():
            if current_time - t_hist.get('last_seen', current_time) > 10.0:
                expired_tids.append(tid)
        
        for tid in expired_tids:
            t_hist = track_history[tid]
            if t_hist['is_staff']:
                if 'staff_id' in t_hist:
                    update_staff_exit(t_hist['staff_id'])
                    staff_name = t_hist.get('staff_name')
                    if staff_name in state.staff_presence:
                        del state.staff_presence[staff_name]
            else:
                if 'customer_id' in t_hist:
                    update_customer_exit(t_hist['customer_id'])
            
            del track_history[tid]
            logging.info(f"Track {tid} expired and removed from history.")

        with state.results_lock:
            state.draw_results = temp_results
            state.new_recognitions_flag = True
