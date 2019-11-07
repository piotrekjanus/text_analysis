from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Conv1D
from keras.optimizers import Adam
import numpy as np
import pandas as pd
from keras import backend as K


from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


data = pd.read_csv('single-5-emb.tsv', sep='\t', header=None).to_numpy()
labels = pd.read_csv('single-5-meta.tsv', sep='\t')

labels = labels['zawod'].to_numpy()

encoder = LabelEncoder()
encoder.fit(labels)
labels = encoder.transform(labels)
from keras.utils import to_categorical
labels = to_categorical(labels)
indices = np.random.permutation(data.shape[0])
size = int(data.shape[0] * 0.8)
id_t, id_ts = indices[:size], indices[size:]
x_train, x_test = data[id_t, :], data[id_ts, :]
y_train, y_test = labels[id_t], labels[id_ts]


callbacks = [EarlyStopping(monitor='val_loss', patience=10)]
             
model = Sequential()

model.add(Dense(256, activation='relu', input_dim=x_train.shape[1]))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy',f1_m])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=400, callbacks=callbacks)

preds = model.predict(x_test)
preds = preds.argmax(axis=1)
y_test = y_test.argmax(axis=1)
a = encoder.inverse_transform(preds)
b = encoder.inverse_transform(y_test)

np.sum(a==b)/len(a)