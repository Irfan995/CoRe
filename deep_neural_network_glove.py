import sys 
import pandas as pd
import numpy as np
import string
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection  import train_test_split
import os, logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras import Sequential, Model
from keras.layers import Embedding, SpatialDropout1D, Conv1D, BatchNormalization, GlobalMaxPool1D, Dropout, Dense, SimpleRNN, LSTM, Bidirectional, Input, GRU, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate 
from tensorflow.keras.metrics import AUC
from tensorflow.keras import backend as K
from sklearn.metrics import f1_score
# import tensorflow_addons as tfa
# import matplotlib.pyplot as plt
from utils.helpers.methods import get_device, get_datetime_str
import time

logging.basicConfig(filename='deep_learning_glove_models' + '_' + get_datetime_str()  + '.log', level=logging.INFO)
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


#Cleaning
totalContentCleaned = []
punctDict = {}
for punct in string.punctuation:
    punctDict[punct] = None
transString = str.maketrans(punctDict)
for sen in df['Request']:
    p = sen.translate(transString)
    totalContentCleaned.append(p)
df['clean_text'] = totalContentCleaned


#Label Binarization
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

#embedding parameters
num_words = 20000 
max_features = 10000 
max_len = 200 
embedding_dims = 128 
num_epochs = 50 
val_split = 0.1
batch_size2 = 128 
sequence_length = 250


#Tokenization
tokenizer = Tokenizer(num_words)
tokenizer.fit_on_texts(list(X_train))

#Convert tokenized sentence to sequnces
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
 
# padding the sequences
X_train = pad_sequences(X_train, max_len)
X_test  = pad_sequences(X_test,  max_len)

X_tra, X_val, y_tra, y_val = train_test_split(X_train, y_train, train_size =0.7, random_state=233)

#Early stopping to stop overfitting
early = EarlyStopping(monitor="val_loss", mode="auto", patience=4)


#Assign weights to contexts
class_frequencies = np.sum(y_train, axis=0)
total_samples = len(y_train)
class_weights = total_samples / (len(class_frequencies) * class_frequencies)
class_weights_dict = {idx: weight for idx, weight in enumerate(class_weights)}


#GloVe
glove_file = open(r'datasets\glove.6B\glove.6B.100d.txt', encoding="utf8")
embeddings_index = dict()
for line in glove_file:
    val = line.split(' ')
    word = val[0]
    coefs = np.asarray(val[1:], dtype='float32')
    embeddings_index[word] = coefs
glove_file.close()

print('Loaded %s word vectors.' % len(embeddings_index))

# Prepare the embedding matrix vectors in order to feed/pass the neural network
embedding_matrix = np.zeros((len(tokenizer.word_index)+1, 100))

for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


class F1ScoreCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        y_pred = (self.model.predict(X_val) > 0.5).astype("int32")
        f1 = f1_score(y_val, y_pred, average='weighted')
        print(f"Weighted F1 Score for epoch {epoch + 1}: {f1}")

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
f1_callback = F1ScoreCallback()



#Convolutional Neural Network (CNN)
logger.info("\n\n-----------------------CNN---------------------\n\n")
CNN_Glove_model = Sequential([
    Embedding(input_dim =embedding_matrix.shape[0], input_length=max_len, output_dim=embedding_matrix.shape[1],weights=[embedding_matrix], trainable=False),
    SpatialDropout1D(0.5),
    Conv1D(filters=100, kernel_size=4, padding='same', activation='relu'),
    BatchNormalization(),
    GlobalMaxPool1D(),
    Dropout(0.5),
    Dense(50, activation = 'relu'),
    Dense(numOfClasses, activation = 'sigmoid')
])

start_time = time.time()
CNN_Glove_model.compile(loss='binary_crossentropy', optimizer="adam", metrics=[f1_score_metric,'binary_accuracy','accuracy', AUC()])
CNN_Glove_model.summary()

CNN_Glove_model_fit = CNN_Glove_model.fit(X_tra, y_tra, batch_size=batch_size2, epochs=num_epochs, validation_data=(X_val, y_val), callbacks=[early, f1_callback, ModelLogger()])
end_time = time.time()

cnn_training = end_time - start_time
logger.info(f"CNN Training: {cnn_training} seconds")

# # Plot training & validation accuracy values
# plt.plot(CNN_Glove_model_fit.history['binary_accuracy'])
# plt.plot(CNN_Glove_model_fit.history['val_binary_accuracy'])
# plt.title('CNN-Glove Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='upper left')
# plt.show()

# # Plot training & validation loss values
# plt.plot(CNN_Glove_model_fit.history['loss'])
# plt.plot(CNN_Glove_model_fit.history['val_loss'])
# plt.title('CNN-Glove Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Training Loss', 'Validation Loss'], loc='lower right')
# plt.show()

#Recurrent Neural Network (RNN)
logger.info("\n\n-----------------------RNN---------------------\n\n")
RNN_Glove_model = Sequential([
    Embedding(input_dim =embedding_matrix.shape[0], input_length=max_len, output_dim=embedding_matrix.shape[1],weights=[embedding_matrix], trainable=False),
    SpatialDropout1D(0.5),
    SimpleRNN(25, return_sequences=True),
    BatchNormalization(),
    Dropout(0.5),
    GlobalMaxPool1D(),
    Dense(50, activation = 'relu'),
    Dense(numOfClasses, activation = 'sigmoid')
])

start_time = time.time()
RNN_Glove_model.compile(loss='binary_crossentropy', optimizer="adam", metrics=[f1_score_metric,'binary_accuracy','accuracy', AUC()])
RNN_Glove_model.summary()

RNN_Glove_model_fit = RNN_Glove_model.fit(X_tra, y_tra, batch_size=batch_size2, epochs=num_epochs, validation_data=(X_val, y_val), callbacks=[early, f1_callback, ModelLogger()])
end_time = time.time()

rnn_training = end_time - start_time
logger.info(f"RNN Training: {rnn_training} seconds")
# # Plot training & validation accuracy values
# plt.plot(RNN_Glove_model_fit.history['binary_accuracy'])
# plt.plot(RNN_Glove_model_fit.history['val_binary_accuracy'])
# plt.title('RNN-Glove Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='upper left')
# plt.show()

# # Plot training & validation loss values
# plt.plot(RNN_Glove_model_fit.history['loss'])
# plt.plot(RNN_Glove_model_fit.history['val_loss'])
# plt.title('RNN-Glove Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Training Loss', 'Validation Loss'], loc='lower right')
# plt.show()


#Long Short Term Memory (LSTM)
logger.info("\n\n-----------------------LSTM---------------------\n\n")
LSTM_Glove_model = Sequential([
    Embedding(input_dim =embedding_matrix.shape[0], input_length=max_len, output_dim=embedding_matrix.shape[1],weights=[embedding_matrix], trainable=False),
    SpatialDropout1D(0.5),
    LSTM(25, return_sequences=True),
    BatchNormalization(),
    Dropout(0.5),
    GlobalMaxPool1D(),
    Dense(50, activation = 'relu'),
    Dense(numOfClasses, activation = 'sigmoid')
])

start_time = time.time()
LSTM_Glove_model.compile(loss='binary_crossentropy', optimizer="adam", metrics=[f1_score_metric,'binary_accuracy','accuracy', AUC()])
LSTM_Glove_model.summary()

LSTM_Glove_model_fit = LSTM_Glove_model.fit(X_tra, y_tra, batch_size=batch_size2, epochs=num_epochs, validation_data=(X_val, y_val), callbacks=[early, f1_callback, ModelLogger()])
end_time = time.time()

lstm_training = end_time - start_time
logger.info(f"LSTM Training: {lstm_training} seconds")
# # Plot training & validation accuracy values
# plt.plot(LSTM_Glove_model_fit.history['binary_accuracy'])
# plt.plot(LSTM_Glove_model_fit.history['val_binary_accuracy'])
# plt.title('LSTM-Glove Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='upper left')
# plt.show()

# # Plot training & validation loss values
# plt.plot(LSTM_Glove_model_fit.history['loss'])
# plt.plot(LSTM_Glove_model_fit.history['val_loss'])
# plt.title('LSTM-Glove Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Training Loss', 'Validation Loss'], loc='lower right')
# plt.show()


#Bidirectional Long Short-Term Memory (BiLSTM)
logger.info("\n\n-----------------------BiLSTM---------------------\n\n")
Bi_LSTM_Glove_model = Sequential([
    Embedding(input_dim =embedding_matrix.shape[0], input_length=max_len, output_dim=embedding_matrix.shape[1],weights=[embedding_matrix], trainable=False),
    SpatialDropout1D(0.5),
    Bidirectional(LSTM(25, return_sequences=True)), 
    BatchNormalization(),
    Dropout(0.5),
    GlobalMaxPool1D(),
    Dense(50, activation = 'relu'),
    Dense(numOfClasses, activation = 'sigmoid')
])

start_time = time.time()
Bi_LSTM_Glove_model.compile(loss='binary_crossentropy', optimizer="adam", metrics=[f1_score_metric,'binary_accuracy','accuracy', AUC()])
Bi_LSTM_Glove_model.summary()

Bi_LSTM_Glove_model_fit = Bi_LSTM_Glove_model.fit(X_tra, y_tra, batch_size=batch_size2, epochs=num_epochs, validation_data=(X_val, y_val), callbacks=[early, f1_callback, ModelLogger()])
end_time = time.time()

bi_lstm_training = end_time - start_time
logger.info(f"BiLSTM Training: {bi_lstm_training} seconds")

# # Plot training & validation accuracy values
# plt.plot(Bi_LSTM_Glove_model_fit.history['binary_accuracy'])
# plt.plot(Bi_LSTM_Glove_model_fit.history['val_binary_accuracy'])
# plt.title('Bidirecitonal LSTM-Glove Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='upper left')
# plt.show()

# # Plot training & validation loss values
# plt.plot(Bi_LSTM_Glove_model_fit.history['loss'])
# plt.plot(Bi_LSTM_Glove_model_fit.history['val_loss'])
# plt.title('Bidirecitonal LSTM-Glove Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Training Loss', 'Validation Loss'], loc='lower right')
# plt.show()

#Gated Recurrent (GRU)
logger.info("\n\n-----------------------GRU---------------------\n\n")
sequence_input = Input(shape=(max_len, ))
GRU_Glove_model = Embedding(input_dim =embedding_matrix.shape[0], input_length=max_len, output_dim=embedding_matrix.shape[1],weights=[embedding_matrix], trainable=False)(sequence_input)
GRU_Glove_model = SpatialDropout1D(0.2)(GRU_Glove_model)
GRU_Glove_model = GRU(128, return_sequences=True,dropout=0.1,recurrent_dropout=0.1)(GRU_Glove_model)
GRU_Glove_model = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(GRU_Glove_model)
avg_pool = GlobalAveragePooling1D()(GRU_Glove_model)
max_pool = GlobalMaxPooling1D()(GRU_Glove_model)
GRU_Glove_model = concatenate([avg_pool, max_pool]) 
preds = Dense(numOfClasses, activation="sigmoid")(GRU_Glove_model)
GRU_Glove_model = Model(sequence_input, preds)
GRU_Glove_model.compile(loss='binary_crossentropy',optimizer="adam",metrics=[f1_score_metric,'binary_accuracy','accuracy', AUC() ])
GRU_Glove_model.summary()

start_time = time.time()
GRU_Glove_model_fit = GRU_Glove_model.fit(X_tra, y_tra, batch_size=batch_size2, epochs=num_epochs, validation_data=(X_val, y_val), callbacks=[early, f1_callback, ModelLogger()])
end_time = time.time()

gru_training = end_time - start_time
logger.info(f"GRU Training: {gru_training} seconds")
# # Plot training & validation accuracy values
# plt.plot(GRU_Glove_model_fit.history['binary_accuracy'])
# plt.plot(GRU_Glove_model_fit.history['val_binary_accuracy'])
# plt.title('Gated Recurrent Unit (GRU) with Glove Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='upper left')
# plt.show()

# # Plot training & validation loss values
# plt.plot(GRU_Glove_model_fit.history['loss'])
# plt.plot(GRU_Glove_model_fit.history['val_loss'])
# plt.title('Bidirectional Gated Recurrent Unit (GRU) with Glove Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Training Loss', 'Validation Loss'], loc='lower right')
# plt.show()
