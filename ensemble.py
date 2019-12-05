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
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import copy

# from read_embeddings import get_labels_and_embeddings, get_onet_train_test_data

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
    print(classification_report(y, preds))
from keras.layers import BatchNormalization
def make_MLP(in_shape, out_shape):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=in_shape))
    model.add(Dropout(0.2))
    # model.add(BatchNormalization())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(out_shape, activation='softmax'))
    return model

def load_data(set_type, file_name):
    data = pd.read_csv('out/{}/{}-emb.tsv'.format(set_type, file_name), sep='\t', header=None).to_numpy()
    labels = pd.read_csv('out/{}/{}-meta.tsv'.format(set_type, file_name), sep='\t')
    labels = labels['zawod'].to_numpy()
    return data, labels

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

def preprocess_data(data, labels):
    ind = np.where(labels == 'politycy')[0]
    # ind = np.random.choice(ind , int(ind.shape[0] * 0.8),replace=False)
    data = np.delete(data, ind, axis=0)
    labels = np.delete(labels, ind, axis=0)
    encoder = LabelEncoder()
    encoder = encoder.fit(labels)
    labels = encoder.transform(labels)
    return data, labels, encoder

def generator(X, Y, batch_size=100):
    while True: 
        idx = []
        var_list = {}
        
        for y in np.unique(Y):
            idd = np.where(Y == y)[0]
            idx.extend(np.random.choice(idd, int(batch_size/np.unique(Y).shape[0]), replace=False))
            var_list[y] = np.var(x_train[idd], axis=0)

        batch_samples = np.array(X)[idx]
        Y_ = np.array(Y)[idx]
        X_train = []
        y_train = []
        for i, batch_sample in enumerate(batch_samples):
            #Do "augmentation"
            if np.random.random() > 0.5:
                batch_sample += np.random.choice([-1,1]) * 0.1*var_list[Y[i]]
            X_train.append(batch_sample)
            y_train.append(Y_[i])

        shuffle = np.random.choice(range(batch_size), batch_size, replace = False)   
        # Make sure they are numpy arrays
        X_train = np.array(X_train)[shuffle]
        y_train = np.array(y_train)[shuffle]
         
        yield X_train, to_categorical(y_train)

## Model to put on the top of ensemble 
def make_one_to_rule_them_all(predictions, real_val):
    x_train, x_test, y_train, y_test = split_dataset(predictions, real_val)
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=predictions.shape[1]))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(real_val.shape[1], activation='softmax'))
    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

    callbacks = [EarlyStopping(monitor='val_loss', patience=10)]
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=80, batch_size=100, callbacks=callbacks)
    return model 

if __name__ == "__main__":
    
    x_train, y_train = load_data('train', 'single-5')
    x_test, y_test = load_data('test', 'single-5')
    x_train, y_train, encoder = preprocess_data(x_train, y_train)
    ind = np.where(y_test == 'politycy')[0] 
    #remove all politicians
    x_test = np.delete(x_test, ind, axis=0)
    y_test = np.delete(y_test, ind, axis=0)

    y_test = encoder.transform(y_test)

    ## get weights of each class if you want to include them in model
    # weights = Counter(y_train)
    # for w in weights.keys():
    #     weights[w] = 1/weights[w] * 400
    ensemble = {}
    ## make MLP
    for lab in list(set(y_train)):
        model = make_MLP(x_train.shape[1], 9) # Make same MLPs, can possibly be changed in every iteration
        model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy',f1_m])
        callbacks = [EarlyStopping(monitor='val_loss', patience=100)]

        # Fit for different types ex. with generator or weighted classes
        model.fit(x_train, to_categorical(y_train), validation_data=(x_test, to_categorical(y_test)),  epochs=15, batch_size=100, callbacks=callbacks)
        # model.fit_generator(generator(x_train, y_train, batch_size=100), validation_data = (x_test, to_categorical(y_test)), epochs=20, steps_per_epoch = 30)
        # model.fit(x_train, to_categorical(y_train),validation_data=(x_test,to_categorical(y_test)), epochs=10, class_weight=weights, batch_size=64 ,callbacks=callbacks)
        ensemble[lab] = model
    ## predict
    # results = []
    # for mod in ensemble.keys():
    #     res = ensemble[mod].predict(x_train)
    #     results.append(res)
    # preds = pd.DataFrame(np.array(results).mean(axis=0)).to_numpy()

    # ultra_model = make_one_to_rule_them_all(preds, to_categorical(y_train))
    
    real_ans = encoder.inverse_transform(y_test)
    ## for soft voting
    results = []
    for mod in ensemble.keys():
        res = ensemble[mod].predict(x_test)
        finito_beng =  encoder.inverse_transform(res.argmax(axis=1))
        results.append(res)
        
    res = pd.DataFrame(np.array(results).mean(axis=0)).to_numpy()
    res = model.predict(x_test)
    finial_answer =  encoder.inverse_transform(res.argmax(axis=1))
    print(classification_report(real_ans, finial_answer))
    print(confusion_matrix(real_ans, finial_answer))
    

