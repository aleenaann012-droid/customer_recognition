import numpy as np

known_faces = []

def compare_face(new_embedding):

    for idx, emb in enumerate(known_faces):

        similarity = np.dot(new_embedding, emb)

        if similarity > 0.5:
            return idx

    known_faces.append(new_embedding)

    return len(known_faces)-1