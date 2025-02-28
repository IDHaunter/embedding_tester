import os
import time
import numpy as np
import fasttext
import fasttext.util
from sklearn.metrics.pairwise import cosine_similarity

# Доступные размеры моделей FastText
MODEL_SIZES = [50, 100, 200, 300]  # Размерности векторов FastText

# Папка для кэша моделей FastText
FASTTEXT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fasttext_cache')
os.makedirs(FASTTEXT_CACHE_DIR, exist_ok=True)


def load_fasttext_model(dim):
    """Загружает модель FastText с указанной размерностью."""
    fasttext.util.download_model('en', if_exists='ignore')  # Загружаем модель, если её нет
    fasttext.util.reduce_model(fasttext.load_model(f"cc.en.300.bin"), dim)
    return fasttext.load_model(f"cc.en.{dim}.bin")


def get_embedding(model, text):
    """Получает эмбеддинг для заданного текста."""
    return np.array(model.get_sentence_vector(text))


def cosine_similarity_score(vec1, vec2):
    """Вычисляет косинусное сходство между двумя векторами."""
    return float(cosine_similarity([vec1], [vec2])[0][0])


def main():
    text1 = "Automobile"
    text2 = "a car"

    print(f"Phrase #1: {text1}")
    print(f"Phrase #2: {text2}\n")

    print("Testing FastText models with different dimensions...\n")

    for dim in MODEL_SIZES:
        print(f"Using FastText model with dimension: {dim}")

        try:
            start_time = time.perf_counter()
            model = load_fasttext_model(dim)
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
