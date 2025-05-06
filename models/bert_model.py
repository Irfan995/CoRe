from transformers import BertTokenizer, BertForSequenceClassification
from data.dataset_loader import Dataset

class BERTBase:
    def __init__(self, model_name: str, dataset: Dataset):
        self.dataset = dataset
        self.model_name = model_name
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=dataset.get_label_length())
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
    
    def get_model(self):
        return self.model
    
    def get_tokenizer(self):
        return self.tokenizer