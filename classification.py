from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Conv1D
from keras.optimizers import Adam
import numpy as np
import pandas as pd
from keras import backend as K
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from collections import Counter
from sklearn.metrics import classification_report
from read_embeddings import get_labels_and_embeddings, get_onet_train_test_data

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

def split_dataset(dat, lbs, ratio = 0.8):
    indices = np.random.permutation(dat.shape[0])
    size = int(dat.shape[0] * ratio)
    id_t, id_ts = indices[:size], indices[size:]
    x_train, x_test = dat[id_t, :], dat[id_ts, :]
    y_train, y_test = lbs[id_t], lbs[id_ts]
    return x_train, x_test, y_train, y_test

def make_predictions(x, y, model):
    preds = model.predict(x)
    preds = preds.argmax(axis=1)
    y = y.argmax(axis=1)
    a = encoder.inverse_transform(preds)
    b = encoder.inverse_transform(y)
    print(classification_report(b, a))

def make_MLP(in_shape, out_shape):
    model = Sequential()
    model.add(Dense(256, activation='relu', input_dim=in_shape))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(out_shape, activation='softmax'))
    return model

def load_data(file_name):
    data = pd.read_csv('{}-emb.tsv'.format(file_name), sep='\t', header=None).to_numpy()
    labels = pd.read_csv('{}-meta.tsv'.format(file_name), sep='\t')
    labels = labels['zawod'].to_numpy()
    return data, labels

def preprocess_data(data, labels):
    ind = np.where(labels == 'politycy')[0]
    ind = np.random.choice(ind , int(ind.shape[0] * 0.85),replace=False)
    data = np.delete(data, ind, axis=0)
    labels = np.delete(labels, ind, axis=0)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)
    labels = to_categorical(labels)
    return data, labels, encoder


def get_onet_data():
    test, train = get_onet_train_test_data(3, 'corpus')
    test = get_labels_and_embeddings(test)
    train = get_labels_and_embeddings(train)
    encoder = LabelEncoder()
    train['labels'] = encoder.fit_transform(train['labels'])
    test['labels'] = encoder.transform(test['labels'])
    train['labels'] = to_categorical( train['labels'])
    test['labels'] = to_categorical( test['labels'])

    return train['embeddings'], test['embeddings'], train['labels'], test['labels']

if __name__ == "__main__":
    
    # data, labels = load_data('corpus-2')
    # data, labels, encoder = preprocess_data(data, labels)
    # x_train, x_test, y_train, y_test = split_dataset(data, labels)

    x_train, x_test, y_train, y_test = get_onet_data()

    ## make MLP
    model = make_MLP(x_train.shape[1], y_train.shape[1])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1_m])
    callbacks = [EarlyStopping(monitor='val_loss', patience=100)]
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=64, callbacks=callbacks)

    ## predict
    make_predictions(x_test, y_test, model)
