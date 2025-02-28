import time
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from sklearn.metrics.pairwise import cosine_similarity

# Указываем модель для тестирования
MODEL_NAME = "BAAI/bge-m3"

def get_embedding(model, text):
    embeddings = model.encode([text], batch_size=1, max_length=8192)['dense_vecs']
    return np.array(embeddings[0])

def cosine_similarity_score(vec1, vec2):
    return float(cosine_similarity([vec1], [vec2])[0][0])

def main():
    prefix = ''  # можно задать "query: " или "document: "
    text1 = f"{prefix}хочу кредит"
    text2 = f"{prefix}кредит нужен мне"

    print(f"Phrase #1: {text1}")
    print(f"Phrase #2: {text2}")

    if not text1 or not text2:
        print("Error: empty phrase entered!")
        return

    print(f"\nUsing model: {MODEL_NAME}\n")

    try:
        model = BGEM3FlagModel(MODEL_NAME, use_fp16=True)
        print("Model loaded successfully.")

        start_time = time.perf_counter()

        vec1 = get_embedding(model, text1)
        vec2 = get_embedding(model, text2)
        similarity = cosine_similarity_score(vec1, vec2)

        execution_time = time.perf_counter() - start_time

        print(f"  Embedding size: {vec1.shape}")
        print(f"  Cosine similarity: {similarity:.4f}")
        print(f"  Calculation time: {execution_time:.6f} sec.\n")

    except Exception as e:
        print(f"  Error: {e}\n")

if __name__ == "__main__":
    main()
