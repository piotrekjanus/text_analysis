from read_embeddings import get_onet_train_test_data, get_labels_and_embeddings
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

import numpy as np

def acc(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


if __name__ == '__main__':

    train, test = get_onet_train_test_data(5, 'corpus')
    train = get_labels_and_embeddings(train)
    test = get_labels_and_embeddings(test)

    enc = LabelEncoder()
    test['labels'] = enc.fit_transform(test['labels'])
    train['labels'] = enc.transform(train['labels'])

    model = xgb.XGBClassifier(objective="multi:softmax")
    model.fit(train['embeddings'], train['labels'])
    print(model)

    predictions = model.predict(test['embeddings'])
    print(accuracy_score(test['labels'], predictions))
    print(f1_score(test['labels'], predictions, average='micro'))
    print(f1_score(test['labels'], predictions, average='macro'))

