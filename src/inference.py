import torch
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer
from tensorflow.keras import backend as K
from tensorflow import keras

@keras.utils.register_keras_serializable()
def f1_score_metric(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision_value = precision(y_true, y_pred)
    recall_value = recall(y_true, y_pred)
    f1_val = 2 * ((precision_value * recall_value) / (precision_value + recall_value + K.epsilon()))
    return f1_val

custom_objects = {
        'f1_score_metric': f1_score_metric
    }
class Inference:
    def __init__(self, model_path: str):
        self.model = load_model(model_path, custom_objects=custom_objects)

    def predict(self, text: str):
        self.model.eval()
        encoding = self.tokenizer(text, truncation=True, padding=True, return_tensors='pt')
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)
            class_names = self.label_encoder.classes_

        predict = preds.cpu().tolist()

        return class_names[predict]

    def predict_sbert(self, text: str, encoder, class_names):
        text_list = []
        text_list.append(text)
        encoded_text = encoder.encode(text_list)
        prediction_list = self.model.predict(encoded_text)[0]
        print(prediction_list)

        predicted_class_dict = {}
        print(class_names)

        for i in range(0, len(class_names)):
            predicted_class_dict[class_names[i]] = prediction_list[i]
        
        for key in predicted_class_dict:
            print(f"Class - {key}: {round(predicted_class_dict[key]*100, 2)}")

# Example usage
if __name__ == "__main__":
    classes = ['Assignments', 'Comfort', 'Frustration', 'Grades' ,'Light', 'Materials',
 'Noise', 'Study Resource', 'Temperature', 'Weather', 'direction' ,'feedback',
 'reminder', 'schedule']
    inference = Inference(r'C:\Users\fai94s\Documents\Context Re Struct\trained_model\saf\CDS\all-MiniLM-L12-v2\saf_model.keras')
    text ="Shut up"
    prediction = inference.predict_sbert(text, SentenceTransformer('all-MiniLM-L12-v2'), classes)
    
