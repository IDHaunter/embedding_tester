import os
import time
import numpy as np
from gensim.models.fasttext import load_facebook_vectors
from sklearn.metrics.pairwise import cosine_similarity

# Доступные размеры моделей FastText (должны быть заранее сохранены!)
MODEL_SIZES = [50]   # [50, 100, 200, 300]

# Папка для хранения моделей FastText
FASTTEXT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fasttext_cache")
os.makedirs(FASTTEXT_CACHE_DIR, exist_ok=True)


def load_fasttext_model(dim):
    """Загружает модель FastText с указанной размерностью."""
    path = os.path.join(FASTTEXT_CACHE_DIR, f"cc.en.{dim}.bin")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Модель {path} отсутствует! Сначала уменьшите и сохраните её.")
    return load_facebook_vectors(path)


def get_embedding(model, text):
    """Получает эмбеддинг для заданного текста."""
    return np.array(model[text])


def cosine_similarity_score(vec1, vec2):
    """Вычисляет косинусное сходство между двумя векторами."""
    return float(cosine_similarity([vec1], [vec2])[0][0])


def main():
    text1 = "bank card"
    text2 = "card bank"

    print(f"Phrase #1: {text1}")
    print(f"Phrase #2: {text2}\n")

    print("Testing FastText models with different dimensions...\n")

    for dim in MODEL_SIZES:
        print(f"Using FastText model with dimension: {dim}")

        try:
            model = load_fasttext_model(dim)
            start_time = time.perf_counter()
            vec1 = get_embedding(model, text1)
            vec2 = get_embedding(model, text2)
            similarity = cosine_similarity_score(vec1, vec2)
            execution_time = time.perf_counter() - start_time

            print(f"  Cosine similarity: {similarity:.4f}")
            print(f"  Calculation time: {execution_time:.6f} sec.\n")

        except Exception as e:
            print(f"  Error: {e}\n")


if __name__ == "__main__":
    main()