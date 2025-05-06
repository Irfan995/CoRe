import pathlib
import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer

class Dataset:
    def __init__(self, file_path: str):
        self.data = self.load_data(file_path)
        self.texts = self.data['Request']
        self.context_labels = self.data['Context']

    def get_label_length(self):
        return len(self.context_labels.unique())

    def load_data(self, file_path: str):
        file_type = pathlib.Path(file_path).suffix
        if file_type == ".csv":
            return pd.read_csv(file_path)
        elif file_type == ".xlsx":
            return pd.read_excel(file_path)
        elif file_type == ".xls":
            return pd.read_excel(file_path)
        else:
            raise ValueError("Invalid file type specified")
        
    def split_data(self, tokenizer: BertTokenizer=None, sbert: SentenceTransformer=None,  test_size: float = 0.2):
        texts = self.data['Request']
        context_labels = self.data['Context']

        # Split the data into training and test sets
        if tokenizer:
            # Encode labels
            label_encoder = LabelEncoder()
            context_labels = label_encoder.fit_transform(context_labels)
            train_texts, test_texts, train_labels, test_labels = train_test_split(texts, context_labels, test_size=test_size, random_state=42)

            # Tokenize and convert text data to input tensors
            train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, return_tensors='pt')
            test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, return_tensors='pt')

            # Convert labels to tensors with the correct data type
            train_labels = torch.tensor(train_labels, dtype=torch.long)
            test_labels = torch.tensor(test_labels, dtype=torch.long)        

            # Create DataLoader for training and testing sets
            train_dataset = TensorDataset(train_encodings.input_ids, train_encodings.attention_mask, train_labels)
            test_dataset = TensorDataset(test_encodings.input_ids, test_encodings.attention_mask, test_labels)

            return train_dataset, test_dataset
        elif sbert:
            # Label Binarization
            label_encoder = LabelBinarizer()
            labels = label_encoder.fit_transform(self.data['Context'])
            total_classes = len(label_encoder.classes_)

            # Coverting to floating value
            label_df = pd.DataFrame(labels, columns=label_encoder.classes_)
            np.asarray(label_df).astype('float32').reshape((-1,1))

            # Encoding voice sample to feed in the network
            voice_list = texts.to_numpy()
            sentence_embeddings = sbert.encode(voice_list)

            # Split dataset into train and test
            train_x, test_x, train_y, test_y = train_test_split(sentence_embeddings, 
                                                      label_df, 
                                                      train_size=0.7, 
                                                      test_size=0.3, 
                                                      random_state=42)
            return train_x, test_x, train_y, test_y, total_classes, label_encoder.classes_
        