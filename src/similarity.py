from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
#staff similarity
def calculate_staff_similarity(
        embedding,
        known_embedding):

    return cosine_similarity(
        embedding,
        known_embedding.reshape(1,-1)
    )[0][0]

#customer similarity
def calculate_customer_similarity(embedding, known_embeddings):

    similarities = cosine_similarity(embedding, known_embeddings)

    max_sim = np.max(similarities)

    best_match_idx = np.argmax(similarities)

    return max_sim, best_match_idx