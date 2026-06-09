import queue
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import state
from database import load_staff, insert_staff_attendance, insert_visitor, load_customers, insert_customer, update_customer
from tracker import FaceCentroidTracker
from similarity import calculate_customer_similarity

staff_details, staff_embeddings = load_staff()
customer_details, customer_embeddings = load_customers()

tracker = FaceCentroidTracker(max_disappeared=30, max_distance=150)

from utils import get_age_group

track_history = {}

def recognition_thread():
     import collections
     while True:
        try:
            faces, detections = state.det_queue.get(timeout=1.0)
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

                    staff_name = th['staff_name']
                    if staff_name not in state.staff_presence:
                        try:
                            insert_staff_attendance(staff_details[i]["staff_id"])
                            state.staff_presence[staff_name] = True
                        except Exception as e:
                            print(f"Error inserting staff attendance: {e}")
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
                        update_customer(matched_cid)
                    else:
                        insert_visitor(age_group, final_gender)
                        new_id = insert_customer(final_age, final_gender, th['embeddings'][-1])
                        customer_embeddings.append(th['embeddings'][-1])
                        customer_details.append({"customer_id": new_id})
                        
                    th['processed'] = True
                except Exception as e:
                    print(f"Error inserting customer stats: {e}")
            
            temp_results.append({
                "track_id": matched_tid,
                "box": face.bbox,
                "text": text
            })

        with state.results_lock:
            state.draw_results = temp_results
            state.new_recognitions_flag = True
