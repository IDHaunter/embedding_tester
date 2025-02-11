from fastembed.embedding import TextEmbedding

supported_models = TextEmbedding.list_supported_models()
print("Supported models in fastembed:")
for model in supported_models:
    print("-", model)