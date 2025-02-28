import os
# Set cache folder for fastembed before import
os.environ["FASTEMBED_CACHE_DIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fastembed_cache')

import time
import numpy as np
from fastembed.embedding import TextEmbedding
from sklearn.metrics.pairwise import cosine_similarity

# Available models for testing (the most promising ones have been added)
# https://qdrant.github.io/fastembed/examples/Supported_Models/
MODELS = [
    "BAAI/bge-small-en-v1.5",               # 384 64MB   stable medium quality but may fall in difficult queries or with typos
    "BAAI/bge-base-en-v1.5",                # 768 208MB   + good with typos                   -2x slower
    "BAAI/bge-large-en-v1.5",               # 1024 1,24GB + good with typos +better semantic  -1.3x slower
    "mixedbread-ai/mxbai-embed-large-v1",   # 1024 1,24GB + good with typos +better semantic  -1.3x slower
    "intfloat/multilingual-e5-large",       # 1024 2,24GB Multilingual
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",       # 384 + russian support хочу-не хочу fail
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"        # 768 + russian support
]

def get_embedding(embedder, text):
    embeddings = list(embedder.embed([text]))
    return np.array(embeddings[0])


def cosine_similarity_score(vec1, vec2):
    return float(cosine_similarity([vec1], [vec2])[0][0])


def main():
    prefix = ''  #  query:  document:
    text1 = f"{prefix}Машина"
    text2 = f"{prefix}Автомобиль"

    # I need a loan to buy a vehicle
    # How to get loan for vehicle?
    # Where is my car?
    # Do you like cars?

    print(f"Phrase #1: {text1}")
    print(f"Phrase #2: {text2}")

    if not text1 or not text2:
        print("Error: empty phrase entered!")
        return

    print("\nTesting models...\n")

    for model_name in MODELS:
        print(f"Using model: {model_name}")

        try:
            embedder = TextEmbedding(model_name=model_name, normalize=True, cache_dir=os.environ["FASTEMBED_CACHE_DIR"])
            print(f"Model cache path: {embedder.cache_dir}")

            start_time = time.perf_counter()

            vec1 = get_embedding(embedder, text1)
            vec2 = get_embedding(embedder, text2)
            similarity = cosine_similarity_score(vec1, vec2)
            execution_time = time.perf_counter() - start_time

            print(len(vec1))
            print(f"  Cosine similarity: {similarity:.4f}")
            print(f"  Calculation time: {execution_time:.6f} sec.\n")
            # print(f"  5 embeddings  {text1}: {vec1[:5]}...")

        except Exception as e:
            print(f"  Error: {e}\n")


if __name__ == "__main__":
    main()
