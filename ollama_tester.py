import time
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODELS = ["paraphrase-multilingual:latest", "bge-m3:latest"]  # Список моделей

def get_embedding(model, text):
    """Получает эмбеддинг текста через локальный сервер Ollama."""
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": text, "stream": False},
        headers={"Content-Type": "application/json"}
    )
    response.raise_for_status()
    embedding = np.array(response.json()["embedding"])
    return embedding

def cosine_similarity_score(vec1, vec2):
    """Вычисляет косинусное сходство между двумя эмбеддингами."""
    return float(cosine_similarity([vec1], [vec2])[0][0])

def vector_norm(vec):
    """Вычисляет норму (модуль) вектора."""
    return np.linalg.norm(vec)

def test_model(model_name, text1, text2):
    """Тестирует модель на двух текстах и выводит результаты."""
    print(f"\nUsing model: {model_name}\n")

    try:
        start_time = time.perf_counter()

        vec1 = get_embedding(model_name, text1)
        vec2 = get_embedding(model_name, text2)
        similarity = cosine_similarity_score(vec1, vec2)

        norm1 = vector_norm(vec1)
        norm2 = vector_norm(vec2)

        execution_time = time.perf_counter() - start_time

        print(f"  Embedding size: {vec1.shape}")
        print(f"  Norm of vector 1: {norm1:.4f}")
        print(f"  Norm of vector 2: {norm2:.4f}")
        print(f"  Cosine similarity: {similarity:.4f}")
        print(f"  Calculation time: {execution_time:.6f} sec.\n")

    except requests.exceptions.RequestException as e:
        print(f"  Error: {e}")

def main():
    text1 = "Плыву на корабле"                # Автомобиль   Хочу кредит     Хочу кредит на авто
    text2 = "передвигаюсь на морском судне"    # машина       Не хочу кредит      Мне нужны деньги на машину

    print(f"Phrase #1: {text1}")
    print(f"Phrase #2: {text2}")

    for model in MODELS:
        test_model(model, text1, text2)

if __name__ == "__main__":
    main()
