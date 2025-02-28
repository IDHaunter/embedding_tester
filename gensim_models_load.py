import fasttext
import fasttext.util
import os
import shutil

# Указываем папку для кэша моделей
FASTTEXT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fasttext_cache")
os.makedirs(FASTTEXT_CACHE_DIR, exist_ok=True)

# Путь к полной модели (300-мерная версия)
MODEL_300_PATH = os.path.join(FASTTEXT_CACHE_DIR, "cc.en.300.bin")

# Если файл не существует, скачиваем его
if not os.path.exists(MODEL_300_PATH):
    print("Downloading FastText model...")
    fasttext.util.download_model("en", if_exists="ignore")

    # Перемещаем скачанную модель в нужную директорию
    downloaded_model = "cc.en.300.bin"  # FastText скачивает модель в текущую папку
    if os.path.exists(downloaded_model):
        shutil.move(downloaded_model, MODEL_300_PATH)

# Загружаем оригинальную модель
print("Loading original model...")
model = fasttext.load_model(MODEL_300_PATH)

# Уменьшаем размерность и сохраняем новые модели
for dim in [50, 100, 200]:
    print(f"Reducing model to {dim} dimensions...")
    fasttext.util.reduce_model(model, dim)
    reduced_model_path = os.path.join(FASTTEXT_CACHE_DIR, f"cc.en.{dim}.bin")
    model.save_model(reduced_model_path)
    print(f"Model with {dim} dimensions saved to {reduced_model_path}")

print("✅ Models are ready for use!")
