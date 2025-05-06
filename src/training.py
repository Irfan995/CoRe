import torch, os, shap, transformers
import tensorflow as tf
import logging
import time
from utils.helpers.methods import get_device, get_datetime_str, calculate_layer_input
from torch.utils.data import DataLoader
from transformers import AdamW
from models.bert_model import BERTBase
from sklearn.metrics import f1_score
import tensorflow_addons as tfa
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.pipeline import make_pipeline
import lime
import lime.lime_text
from lime.lime_tabular import LimeTabularExplainer
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging
logger = logging.getLogger()

class ModelLogger(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.now = time.time()
    def on_epoch_end(self, epoch, logs=None): 
        later = time.time()
        duration = later-self.now 
        logger.info(f'Epoch {epoch +1} Duration {str(duration)} seconds : {logs}')

def f1_score_metric(y_true, y_pred):
    def recall(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision_value = precision(y_true, y_pred)
    recall_value = recall(y_true, y_pred)
    f1_val = 2 * ((precision_value * recall_value) / (precision_value + recall_value + K.epsilon()))
    return f1_val

class Trainer:
    def __init__(self, model, tokenizer=None, train_dataset=None, test_dataset=None, eval_location=None, save_model_location=None, batch_size: int = 32, learning_rate: float =2e-5, epochs: int = 10):
        self.model = model
        self.tokenizer = tokenizer
        self.device = get_device()
        self.model.to(self.device)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Load and split the data
        if train_dataset and test_dataset:
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
            self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.epochs = epochs

        self.best_weighted_f1 = 0.0
        self.patience = 1
        self.num_epochs_without_improvement = 0

        self.date_time = get_datetime_str()

        self.eval_location = eval_location
        if not os.path.exists(self.eval_location):
            os.mkdir(self.eval_location)

        self.save_model_location = save_model_location

    def train(self):
        for epoch in range(self.epochs):
            before_epoch_start = time.time()
            self.model.train()
            for batch in self.train_dataloader:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()

            # Evaluation on the training set after each epoch
            self.model.eval()
            train_predictions = []
            train_true_labels = []

            correct = 0
            with torch.no_grad():
                for batch in self.train_dataloader:
                    input_ids, attention_mask, labels = batch
                    input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    _, predicted_labels = torch.max(logits, dim=1)

                    # Calculate training accuracy (optional)
                    correct += (predicted_labels == labels).sum().item()

                    train_predictions.extend(predicted_labels.cpu().numpy())
                    train_true_labels.extend(labels.cpu().numpy())
            train_acc = 100 * correct / len(self.train_dataloader.dataset)
            after_epoch_end = time.time()
            training_duration = after_epoch_end - before_epoch_start
            train_weighted_f1 = f1_score(train_true_labels, train_predictions, average='weighted')
            print(f"Epoch {epoch+1} - Training Weighted F1 Score: {train_weighted_f1} - Acc: {train_acc} - Duration: {training_duration}")
            train_line = f'Epoch {epoch+1}, Batch Size: {self.batch_size}, Learning Rate: {self.learning_rate}, Duration: {training_duration}, Train F1 Score: {train_weighted_f1}, Accuracy: {train_acc}' + '\n'

            test_weighted_f1, test_acc = self.evaluate()
            print(f"Epoch {epoch+1} - Testing Weighted F1 Score: {test_weighted_f1} - Acc: {test_acc}")
            test_line = f'Epoch {epoch+1}, Batch Size: {self.batch_size}, Learning Rate: {self.learning_rate}, Duration: {training_duration}, Test F1 Score: {test_weighted_f1}, Accuracy: {test_acc}' + '\n'


            eval_filename = self.eval_location + '/model_' + self.date_time  + '.txt'
            with open(eval_filename, 'a') as file:
                        file.write(train_line)
                        file.write(test_line)

            # Check for early stopping
            if test_weighted_f1 > self.best_weighted_f1:
                self.best_weighted_f1 = test_weighted_f1
                self.num_epochs_without_improvement = 0
            else:
                self.num_epochs_without_improvement += 1
                if self.num_epochs_without_improvement >= self.patience:
                    print("Early stopping triggered.")
                    break
        
        # Save the trained mode
        if os.path.isdir(self.save_model_location):
            torch.save(self.model.state_dict(), self.save_model_location + '/model_' + self.date_time + '.pt')
        else:
            raise ValueError('Invalid file saving location')
    
    def train_sbert(self, total_classes, train_x, train_y, test_x, test_y, dimension):
        filter, input_layer, hidden_layer_one, hidden_layer_two = calculate_layer_input(dimension)

        # Fully connected layer
        input_layer = Input(shape=(input_layer,))
        hidden_layer_one = Dense(hidden_layer_one, activation="relu")(input_layer)
        hidden_layer_two = Dense(hidden_layer_two, activation="relu")(hidden_layer_one)
        output_layer= Dense(total_classes, activation="sigmoid")(hidden_layer_two)
        
        # Define model
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            optimizer=optimizer,
                    loss='mse', metrics=["accuracy",
                                         "binary_accuracy",
                                         f1_score_metric]
        )

        # Train model
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=2,
                              verbose=0, mode='auto')
        start_time = time.time()
        trained_model=model.fit(train_x, train_y, epochs=20, batch_size=32, validation_data=(test_x, test_y),callbacks=[callback, ModelLogger()])
        end_time = time.time()

        training_duration = start_time - end_time
        logger.info(f"Total Training Duration : {training_duration}")
        model.summary()

        # Save the trained mode
        if os.path.isdir(self.save_model_location):
            model.save(self.save_model_location + '/model' + '.keras')
        else:
            raise ValueError('Invalid file saving location')
        
    def train_saf(self, total_classes, train_x, train_y, test_x, test_y, dimension, vocab):
        filter, input_layer, hidden_layer_one, hidden_layer_two = calculate_layer_input(dimension)

        # Attention Layer - starts
        query_input = tf.keras.Input(shape=(input_layer,), dtype='float32')  # dq
        value_input = tf.keras.Input(shape=(input_layer,), dtype='float32')  # dv
        token_embedding = tf.keras.layers.Embedding(input_dim=len(vocab), output_dim=input_layer)  # dk
        query_embeddings = token_embedding(query_input)  # Embedding vector
        value_embeddings = token_embedding(value_input)  # Embedding vector

        # CNN layer as input
        cnn_layer = tf.keras.layers.Conv1D(
            filters=filter,
            kernel_size=4,
            padding='same')  
        query_seq_encoding = cnn_layer(query_embeddings)
        value_seq_encoding = cnn_layer(value_embeddings)
        query_value_attention_seq = tf.keras.layers.Attention()([query_seq_encoding, value_seq_encoding])  # First MatMul
        query_encoding = tf.keras.layers.GlobalAveragePooling1D()(query_seq_encoding)
        query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(query_value_attention_seq)
        input_layer = tf.keras.layers.Concatenate()([query_encoding, query_value_attention])
        # Attention Layer - ends

        # Fully connected layer
        input_layer = Input(shape=(input_layer,), tensor=input_layer)
        hidden_layer_one = Dense(hidden_layer_one, activation="relu")(input_layer)
        hidden_layer_two = Dense(hidden_layer_two, activation="relu")(hidden_layer_one)
        output_layer= Dense(total_classes, activation="sigmoid")(hidden_layer_two)
        
        # Define model
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.00002)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            optimizer=optimizer,
                    loss='mse', metrics=["accuracy",
                                         "binary_accuracy",
                                         f1_score_metric]
        )

        # Train model
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=5,
                              verbose=0, mode='auto')
        start_time = time.time()
        trained_model=model.fit(train_x, train_y, epochs=20, batch_size=32, validation_data=(test_x, test_y),callbacks=[callback, ModelLogger()])
        end_time = time.time()
        duration = start_time - end_time
        logger.info(f"Total Training Duration: {duration} seconds")
        model.summary()

        # Save the trained mode
        if os.path.isdir(self.save_model_location):
            model.save(self.save_model_location + '/saf_model' + '.keras')
        else:
            raise ValueError('Invalid file saving location')
        
        
    def train_lsaf(self, total_classes, train_x, train_y, test_x, test_y, dimension, vocab):
        filter, input_layer, hidden_layer_one, hidden_layer_two = calculate_layer_input(dimension)

        lstm_input_layer = Input(shape=(None, int(input_layer)))
        lstm_output = LSTM(int(hidden_layer_one), return_sequences=True)(lstm_input_layer)

        print(lstm_output)

        # Attention Layer - starts
        query_input = tf.keras.Input(shape=(input_layer,), dtype='float32')  # dq
        value_input = tf.keras.Input(shape=(input_layer,), dtype='float32')  # dv
        token_embedding = tf.keras.layers.Embedding(input_dim=len(vocab), output_dim=input_layer)  # dk
        query_embeddings = token_embedding(query_input)  # Embedding vector
        value_embeddings = token_embedding(value_input)  # Embedding vector

        # CNN layer as input
        cnn_layer = tf.keras.layers.Conv1D(
            filters=filter,
            kernel_size=4,
            padding='same')  
        query_seq_encoding = cnn_layer(query_embeddings)
        value_seq_encoding = cnn_layer(value_embeddings)
        query_value_attention_seq = tf.keras.layers.Attention()([query_seq_encoding, value_seq_encoding])  # First MatMul
        query_encoding = tf.keras.layers.GlobalAveragePooling1D()(query_seq_encoding)
        query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(query_value_attention_seq)
        input_layer = tf.keras.layers.Concatenate()([query_encoding, query_value_attention])
        # Attention Layer - ends

        # Fully connected layer
        input_layer = Input(shape=(input_layer,), tensor=input_layer)
        hidden_layer_one = Dense(hidden_layer_one, activation="relu")(input_layer)
        hidden_layer_two = Dense(hidden_layer_two, activation="relu")(hidden_layer_one)
        output_layer= Dense(total_classes, activation="sigmoid")(hidden_layer_two)
        
        # Define model
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            optimizer='adam',
                    loss='mse', metrics=["accuracy",
                                         "binary_accuracy",
                                         f1_score_metric]
        )

        # Train model
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=5,
                              verbose=0, mode='auto')
        trained_model=model.fit(train_x, train_y, epochs=20, batch_size=32, validation_data=(test_x, test_y),callbacks=[callback, ModelLogger()])
    
        model.summary()

        # Save the trained mode
        if os.path.isdir(self.save_model_location):
            model.save(self.save_model_location + '/slaf_model' + '.keras')
        else:
            raise ValueError('Invalid file saving location')

    def evaluate(self):
        # Evaluation on the test set after each epoch
        self.model.eval()
        test_predictions = []
        test_true_labels = []
        correct = 0
        with torch.no_grad():
            for batch in self.test_dataloader:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                _, predicted_labels = torch.max(logits, dim=1)
                correct += (predicted_labels == labels).sum().item()

                test_predictions.extend(predicted_labels.cpu().numpy())
                test_true_labels.extend(labels.cpu().numpy())

        test_weighted_f1 = f1_score(test_true_labels, test_predictions, average='weighted')
        test_acc = 100 * correct / len(self.test_dataloader.dataset)
        
        return test_weighted_f1, test_acc


# Example usage
if __name__ == "__main__":
    trainer = Trainer('bert-base-uncased', 'path_to_your_dataset_file')
    trainer.train()
    trainer.evaluate()
