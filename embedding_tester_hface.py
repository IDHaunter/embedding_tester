import os
import time
# import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# Installing the Hugging Face cache folder
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'huggingface_cache')
os.environ["HF_HOME"] = CACHE_DIR

# Loading the model
model_name = "mixedbread-ai/mxbai-embed-large-v1"
model = SentenceTransformer(model_name, cache_folder=CACHE_DIR)

def get_embedding(text, dim=1024, normalize_vectors=True):
    # Getting Text Embedding
    embedding = model.encode([text], convert_to_numpy=True)[0]

    # If you need to reduce the dimensionality, we average
    if dim < embedding.shape[0]:
        embedding = embedding[:dim]  # Cut to the required size

    # Normalization
    if normalize_vectors:
        embedding = normalize([embedding])[0]

    return embedding

def cosine_similarity_score(vec1, vec2):
    # Calculates cosine similarity between vectors
    return float(cosine_similarity([vec1], [vec2])[0][0])

def main():
    text1 = "Good day! Tell me how many people can I take with me to the Minsk1 airport business lounge with my card?"
    text2 = "with a visa card, how many people can i take with me to the business lounge?"

    print(f"Phrase #1: {text1}")
    print(f"Phrase #2: {text2}\n")

    for dim in [1024, 512, 256, 128]:  # dimensions for the test
        print(f"Testing dimension: {dim}")

        try:
            start_time = time.perf_counter()

            vec1 = get_embedding(text1, dim=dim)
            vec2 = get_embedding(text2, dim=dim)
            similarity = cosine_similarity_score(vec1, vec2)
            execution_time = time.perf_counter() - start_time

            print(f"  Cosine similarity: {similarity:.4f}")
            print(f"  Calculation time: {execution_time:.6f} sec.\n")

        except Exception as e:
            print(f"  Error: {e}\n")

if __name__ == "__main__":
    main()
