import os
import time
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# Пути к ONNX-моделям
# wget https://huggingface.co/Xenova/all-MiniLM-L6-v2-onnx/resolve/main/model.onnx -O all-MiniLM-L6-v2.onnx
# all-MiniLM-L6-v2.onnx
# all-MiniLM-L6-v2_quantized.onnx
ONNX_MODELS = {
    "all-MiniLM-L6-v2": "models/all-MiniLM-L6-v2_quantized.onnx",
}

# Загружаем токенизатор
tokenizer = AutoTokenizer.from_pretrained("nixiesearch/all-MiniLM-L6-v2-onnx")


def get_embedding_onnx(onnx_session, text):
    """Получение эмбеддинга через ONNX-модель"""
    tokens = tokenizer(text, return_tensors="np", padding=True, truncation=True)

    input_feed = {
        "input_ids": tokens["input_ids"].astype(np.int64),
        "attention_mask": tokens["attention_mask"].astype(np.int64)
    }

    if "token_type_ids" in tokens:
        input_feed["token_type_ids"] = tokens["token_type_ids"].astype(np.int64)

    # Запуск ONNX-модели
    outputs = onnx_session.run(None, input_feed)

    # 🔍 Отладка: смотрим размерности
    print(f"🔎 Model outputs: {[o.shape for o in outputs]}")

    embeddings = outputs[0]  # Это (1, N, 384), где N — число токенов
    embeddings = embeddings[0]  # Убираем размерность батча (N, 384)

    # Усредняем по токенам
    return embeddings.mean(axis=0)  # Теперь всегда (384,)


def cosine_similarity_score(vec1, vec2):
    """Расчет косинусного сходства"""
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


def test_model(model_name):
    """Тестирование ONNX-модели"""
    print(f"🔹 Тест модели: {model_name} [ONNX]")

    try:
        onnx_session = ort.InferenceSession(ONNX_MODELS[model_name], providers=["CPUExecutionProvider"])
        start_time = time.perf_counter()

        vec1 = get_embedding_onnx(onnx_session, TEXT1)
        vec2 = get_embedding_onnx(onnx_session, TEXT2)

        similarity = cosine_similarity_score(vec1, vec2)
        execution_time = time.perf_counter() - start_time

        print(f"  ✅ Косинусное сходство: {similarity:.4f}")
        print(f"  ⚡ Время расчета: {execution_time:.6f} сек.\n")

    except Exception as e:
        print(f"  ❌ Ошибка: {e}\n")


if __name__ == "__main__":
    # 📝 Входные фразы
    TEXT1 = "smallcat"
    TEXT2 = "small cat"

    print(f"📌 Фраза 1: {TEXT1}")
    print(f"📌 Фраза 2: {TEXT2}\n")

    # 🚀 Тестируем ONNX модели
    for model in ONNX_MODELS:
        test_model(model)
