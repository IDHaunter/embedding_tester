import onnxruntime as ort
import numpy as np
import time
from transformers import AutoTokenizer

# Пути к файлам модели
MODEL_PATH = "models/bge-m3/model.onnx"
TOKENIZER_PATH = "BAAI/bge-m3"  # Hugging Face репозиторий

# Загрузка ONNX модели
ort_session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

# Загрузка токенизатора
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

def encode_text(text, max_length=512):
    """Токенизация текста и создание входных данных для модели"""
    tokens = tokenizer(text, padding="max_length", truncation=True, max_length=max_length, return_tensors="np")

    input_ids = tokens["input_ids"].astype(np.int64)
    attention_mask = tokens["attention_mask"].astype(np.int64)

    return input_ids, attention_mask


def get_embedding(text):
    """Получение усреднённого эмбеддинга для текста"""
    input_ids, attention_mask = encode_text(text)

    # Запуск ONNX модели
    outputs = ort_session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})

    # Первый выход модели — это эмбеддинги (1, 512, 1024)
    embedding = outputs[0]

    # Усредняем эмбеддинги по токенам и убираем лишнюю размерность (1, 1024)
    mean_embedding = embedding.mean(axis=1).squeeze(0)

    return mean_embedding


def cosine_similarity(vec1, vec2):
    """Вычисление косинусного сходства"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


if __name__ == "__main__":
    text1 = "автомобиль"
    text2 = "машина"

    start_time = time.perf_counter()
    emb1 = get_embedding(text1)  # (1024,)
    emb2 = get_embedding(text2)  # (1024,)
    execution_time = time.perf_counter() - start_time

    print(f"- {text1}")
    print(f"- {text2}")
    print(f"Размерность эмбеддингов: {emb1.shape}")
    print(f"Эмбеддинг 1: {emb1[:5]}...")  # Выводим только часть, чтобы не засорять логи
    print(f"Эмбеддинг 2: {emb2[:5]}...")

    # Косинусное сходство
    similarity = cosine_similarity(emb1, emb2)
    print(f"Косинусное сходство: {similarity:.4f}")
    print(f"  ⚡ Время расчета: {execution_time:.6f} сек.\n")
