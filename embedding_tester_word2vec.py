import os
import time
import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

# Папка для кэша моделей Word2Vec
WORD2VEC_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "word2vec_cache")
os.makedirs(WORD2VEC_CACHE_DIR, exist_ok=True)

# Путь к модели
MODEL_PATH = os.path.join(WORD2VEC_CACHE_DIR, "GoogleNews-vectors-negative300.bin")


def download_word2vec_model():
    """Скачивает предобученную модель Word2Vec, если её нет"""
    if not os.path.exists(MODEL_PATH):
        from gensim.downloader import load
        print("Downloading Word2Vec model (GoogleNews-300)...")
        model = load("word2vec-google-news-300")  # Загружаем модель из gensim
        model.save(MODEL_PATH)
        print("✅ Model downloaded and saved.")


def load_word2vec_model():
    """Загружает предобученную модель Word2Vec"""
    print("Loading Word2Vec model...")
    return KeyedVectors.load(MODEL_PATH)


def get_embedding(model, text):
    """Извлекает эмбеддинг из модели (усредняет слова, если их несколько)"""
    words = text.split()
    vectors = [model[word] for word in words if word in model]

    if not vectors:
        raise ValueError(f"⚠️ No words from '{text}' found in the model!")

    return np.mean(vectors, axis=0)  # Усредняем векторы слов


def cosine_similarity_score(vec1, vec2):
    """Вычисляет косинусное сходство"""
    return float(cosine_similarity([vec1], [vec2])[0][0])


def main():
    text1 = "fast credit card"
    text2 = "fast car"

    print(f"Phrase #1: {text1}")
    print(f"Phrase #2: {text2}\n")

    try:
        download_word2vec_model()
        model = load_word2vec_model()

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
