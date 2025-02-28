import time
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Указываем модель для тестирования
MODEL_NAME = "ai-forever/sbert_large_nlu_ru"


def get_embedding(model, tokenizer, text):
    """Получение эмбеддинга для текста"""
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input)

    token_embeddings = model_output.last_hidden_state
    attention_mask = encoded_input['attention_mask']

    # Усреднение эмбеддингов с учетом маски внимания
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    mean_embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),
                                                                                        min=1e-9)

    return mean_embedding.squeeze(0).cpu().numpy()


def cosine_similarity_score(vec1, vec2):
    """Вычисление косинусного сходства"""
    return float(cosine_similarity([vec1], [vec2])[0][0])


def main():
    text1 = "хочу кредит"
    text2 = "не хочу кредит"

    print(f"Phrase #1: {text1}")
    print(f"Phrase #2: {text2}")
    print(f"\nUsing model: {MODEL_NAME}\n")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        print("Model loaded successfully.")

        start_time = time.perf_counter()

        vec1 = get_embedding(model, tokenizer, text1)
        vec2 = get_embedding(model, tokenizer, text2)
        similarity = cosine_similarity_score(vec1, vec2)

        execution_time = time.perf_counter() - start_time

        print(f"  Embedding size: {vec1.shape}")
        print(f"  Cosine similarity: {similarity:.4f}")
        print(f"  Calculation time: {execution_time:.6f} sec.\n")

    except Exception as e:
        print(f"  Error: {e}\n")


if __name__ == "__main__":
    main()
