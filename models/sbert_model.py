# models/sbert_models.py
from sentence_transformers import SentenceTransformer
from data.dataset_loader import Dataset


class SBERTBase:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
    
    def get_model(self):
        return self.model
    
    def get_tokenizer(self):
        return self.model

    def get_vocab(self):
        tokenizer = self.model.tokenizer
        return tokenizer.vocab