import os
from models.bert_model import BERTBase
from models.sbert_model import SBERTBase
from utils.constants.model_constant import BERT_MODEL_NAMES, SBERT_MODEL_NAMES
from data.dataset_loader import Dataset

class ModelLoader:
    def __init__(self, model_type: str, model_name: str, dataset: Dataset):
        self.model_type = model_type
        self.model_name = model_name
        self.dataset = dataset
        
    
    def load_model(self):
        if self.model_type == 'bert':
            return BERTBase(self.model_name, self.dataset)
        elif self.model_type == 'sbert' or self.model_type == 'saf' or self.model_type == 'slaf':
            return SBERTBase(SBERT_MODEL_NAMES[self.model_name]['name'])
        else:
            raise ValueError("Invalid model type specified")

# Example usage
if __name__ == "__main__":
    loader = ModelLoader('bert', 'base-uncased')
    model = loader.load_model()
    print(model.get_model())