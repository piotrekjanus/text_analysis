import pandas as pd
import os
import numpy as np

def load_data(filename):
    valid_labels = ['aktorzy', 'duchowni', 'dziennikarze', 'lekarze', 'malarze', 'muzycy', 'poeci', 'politycy', 'prawnicy', 'sportowcy']
    data = pd.read_csv(f'{filename}', sep=';', names=['entity', 'label', 'embedding'])
    data = data[ data['label'].isin(valid_labels)]
    data['embedding'] = data['embedding'].apply(lambda x: 
                           np.fromstring( x[1:].replace(' ', ''), sep=','))
    return data

def get_data(corpus_dir):
    test_path = os.path.join(corpus_dir, 'Test.csv')
    train_path = os.path.join(corpus_dir, 'Train.csv')
    test_data = load_data(test_path)
    train_data = load_data(train_path)
    return train_data, test_data

def get_onet_train_test_data(window_size, context_lvl):
    corpus_name = f'onet{window_size}_{context_lvl}'
    return get_data(os.path.join('zawody_ONET', corpus_name))

def get_labels_and_embeddings(df):
    return {'labels' : df['label'].to_numpy(), 'embeddings' : np.array(df['embedding'].tolist())}

if __name__ == '__main__':
    train, test = get_onet_train_test_data(5, 'corpus')
    test = get_labels_and_embeddings(test)
    print(test['embeddings'].shape)
