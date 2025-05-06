import sys
import pandas as pd
import numpy as np
import string
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Embedding, SpatialDropout1D, Conv1D, BatchNormalization, GlobalMaxPool1D, Dropout, Dense, SimpleRNN, LSTM, Bidirectional, Input, GRU, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate
from tensorflow.keras.metrics import AUC
from tensorflow.keras import backend as K
from sklearn.metrics import f1_score
import spacy
from utils.helpers.methods import get_device, get_datetime_str
import time
import os, logging


logging.basicConfig(filename='deep_learning_word2vec_models' + '_' + get_datetime_str()  + '.log', level=logging.INFO)
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

# Dataset selection
df = pd.read_excel(r"datasets\classroom_short_commands.xlsx")

# Cleaning
totalContentCleaned = []
punctDict = {}
for punct in string.punctuation:
    punctDict[punct] = None
transString = str.maketrans(punctDict)
for sen in df['Voice']:
    p = sen.translate(transString)
    totalContentCleaned.append(p)
df['clean_text'] = totalContentCleaned


# Label Binarization
multilabel = MultiLabelBinarizer()
y = multilabel.fit_transform(df['Context'])
numOfClasses=len(multilabel.classes_)
y=pd.DataFrame(y, columns=multilabel.classes_)
Y=y[[x for x in multilabel.classes_]]
df=df.reset_index()
Y=Y.reset_index()
df = pd.merge(Y,df,on='index')

# train test split
train, test = train_test_split(df, train_size=0.7, test_size=0.3, random_state=42)

X_train = train["clean_text"]
X_test  = test["clean_text"]
Y=y[[x for x in multilabel.classes_]]
y_train = train[[x for x in multilabel.classes_]]
y_test  = test[[x for x in multilabel.classes_]]

# embedding parameters
num_words = 20000
max_features = 10000
max_len = 200
embedding_dims = 128
num_epochs = 10
val_split = 0.1
batch_size2 = 256
sequence_length = 250

# Tokenization
tokenizer = Tokenizer(num_words)
tokenizer.fit_on_texts(X_train.tolist())

# Convert tokenized sentence to sequences
X_train = tokenizer.texts_to_sequences(X_train.tolist())
X_test = tokenizer.texts_to_sequences(X_test.tolist())

# padding the sequences
X_train = pad_sequences(X_train, max_len)
X_test  = pad_sequences(X_test, max_len)

X_tra, X_val, y_tra, y_val = train_test_split(X_train, y_train, train_size =0.7, random_state=233)

# Early stopping to stop overfitting
early = EarlyStopping(monitor="loss", mode="auto", patience=4)

# Assign weights to contexts
class_frequencies = np.sum(y_train, axis=0)
total_samples = len(y_train)
class_weights = total_samples / (len(class_frequencies) * class_frequencies)
class_weights_dict = {idx: weight for idx, weight in enumerate(class_weights)}

# Load SpaCy model
nlp = spacy.load("en_core_web_md")

# Create embedding matrix using SpaCy vectors
embedding_dim = nlp.vocab.vectors_length
embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))

for word, i in tokenizer.word_index.items():
    if word in nlp.vocab:
        embedding_matrix[i] = nlp.vocab[word].vector

class F1ScoreCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        y_pred = (self.model.predict(X_val) > 0.5).astype("int32")
        f1 = f1_score(y_val, y_pred, average='weighted')
        print(f"Weighted F1 Score for epoch {epoch + 1}: {f1}")

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
f1_callback = F1ScoreCallback()

# Convolutional Neural Network (CNN)
print("\n\n-----------------------CNN---------------------\n\n")
CNN_w2v_model = Sequential([
    Embedding(input_dim =embedding_matrix.shape[0], input_length=max_len, output_dim=embedding_matrix.shape[1],weights=[embedding_matrix], trainable=False),
    SpatialDropout1D(0.5),
    Conv1D(filters=100, kernel_size=4, padding='same', activation='relu'),
    BatchNormalization(),
    GlobalMaxPool1D(),
    Dropout(0.5),
    Dense(50, activation = 'relu'),
    Dense(numOfClasses, activation = 'sigmoid')
])
CNN_w2v_model.compile(loss='binary_crossentropy', optimizer="adam", metrics=[f1_score_metric,'binary_accuracy','accuracy', AUC()])
CNN_w2v_model.summary()

CNN_w2v_model_fit = CNN_w2v_model.fit(X_tra, y_tra, batch_size=batch_size2, epochs=num_epochs, validation_data=(X_val, y_val), callbacks=[early, f1_callback, ModelLogger()])

# Recurrent Neural Network (RNN)
print("\n\n-----------------------RNN---------------------\n\n")
RNN_w2v_model = Sequential([
    Embedding(input_dim =embedding_matrix.shape[0], input_length=max_len, output_dim=embedding_matrix.shape[1],weights=[embedding_matrix], trainable=False),
    SpatialDropout1D(0.5),
    SimpleRNN(25, return_sequences=True),
    BatchNormalization(),
    Dropout(0.5),
    GlobalMaxPool1D(),
    Dense(50, activation = 'relu'),
    Dense(numOfClasses, activation = 'sigmoid')
])
RNN_w2v_model.compile(loss='binary_crossentropy', optimizer="adam", metrics=[f1_score_metric,'binary_accuracy','accuracy', AUC()])
RNN_w2v_model.summary()

RNN_w2v_model_fit = RNN_w2v_model.fit(X_tra, y_tra, batch_size=batch_size2, epochs=num_epochs, validation_data=(X_val, y_val), callbacks=[early, f1_callback, ModelLogger()])

# Long Short Term Memory (LSTM)
print("\n\n-----------------------LSTM---------------------\n\n")
LSTM_w2v_model = Sequential([
    Embedding(input_dim =embedding_matrix.shape[0], input_length=max_len, output_dim=embedding_matrix.shape[1],weights=[embedding_matrix], trainable=False),
    SpatialDropout1D(0.5),
    LSTM(25, return_sequences=True),
    BatchNormalization(),
    Dropout(0.5),
    GlobalMaxPool1D(),
    Dense(50, activation = 'relu'),
    Dense(numOfClasses, activation = 'sigmoid')
])
LSTM_w2v_model.compile(loss='binary_crossentropy', optimizer="adam", metrics=[f1_score_metric,'binary_accuracy','accuracy', AUC()])
LSTM_w2v_model.summary()

LSTM_w2v_model_fit = LSTM_w2v_model.fit(X_tra, y_tra, batch_size=batch_size2, epochs=num_epochs, validation_data=(X_val, y_val), callbacks=[early, f1_callback, ModelLogger()])

# Bidirectional Long Short-Term Memory (BiLSTM)
print("\n\n-----------------------BiLSTM---------------------\n\n")
Bi_LSTM_w2v_model = Sequential([
    Embedding(input_dim =embedding_matrix.shape[0], input_length=max_len, output_dim=embedding_matrix.shape[1],weights=[embedding_matrix], trainable=False),
    SpatialDropout1D(0.5),
    Bidirectional(LSTM(25, return_sequences=True)),
    BatchNormalization(),
    Dropout(0.5),
    GlobalMaxPool1D(),
    Dense(50, activation = 'relu'),
    Dense(numOfClasses, activation = 'sigmoid')
])
Bi_LSTM_w2v_model.compile(loss='binary_crossentropy', optimizer="adam", metrics=[f1_score_metric,'binary_accuracy','accuracy', AUC()])
Bi_LSTM_w2v_model.summary()

Bi_LSTM_w2v_model_fit = Bi_LSTM_w2v_model.fit(X_tra, y_tra, batch_size=batch_size2, epochs=num_epochs, validation_data=(X_val, y_val), callbacks=[early, f1_callback, ModelLogger()])

# Gated Recurrent (GRU)
print("\n\n-----------------------GRU---------------------\n\n")
sequence_input = Input(shape=(max_len, ))
GRU_w2v_model = Embedding(input_dim =embedding_matrix.shape[0], input_length=max_len, output_dim=embedding_matrix.shape[1],weights=[embedding_matrix], trainable=False)(sequence_input)
GRU_w2v_model = SpatialDropout1D(0.2)(GRU_w2v_model)
GRU_w2v_model = GRU(128, return_sequences=True,dropout=0.1,recurrent_dropout=0.1)(GRU_w2v_model)
GRU_w2v_model = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(GRU_w2v_model)
avg_pool = GlobalAveragePooling1D()(GRU_w2v_model)
max_pool = GlobalMaxPooling1D()(GRU_w2v_model)
GRU_w2v_model = concatenate([avg_pool, max_pool])
preds = Dense(numOfClasses, activation="sigmoid")(GRU_w2v_model)
GRU_w2v_model = Model(sequence_input, preds)
GRU_w2v_model.compile(loss='binary_crossentropy',optimizer="adam",metrics=[f1_score_metric,'binary_accuracy','accuracy', AUC() ])
GRU_w2v_model.summary()

GRU_w2v_model_fit = GRU_w2v_model.fit(X_tra, y_tra, batch_size=batch_size2, epochs=num_epochs, validation_data=(X_val, y_val), callbacks=[early, f1_callback, ModelLogger()])