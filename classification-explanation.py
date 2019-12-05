from lime.lime_text import LimeTextExplainer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from read_files import get_embedding, extract_sentences, clear_sentence_and_locate_entities, load_files, list_people, createEmbeddings
import env
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm

def preprocess_data(data, labels):
    ind = np.where(labels == 'politycy')[0]
    # ind = np.random.choice(ind , int(ind.shape[0] * 0.8),replace=False)
    data = np.delete(data, ind, axis=0)
    labels = np.delete(labels, ind, axis=0)
    encoder = LabelEncoder()
    encoder = encoder.fit(labels)
    labels = encoder.transform(labels)
    return data, labels, encoder


def load_data(set_type, file_name):
    data = pd.read_csv('out/{}/{}-emb.tsv'.format(set_type, file_name), sep='\t', header=None).to_numpy()
    labels = pd.read_csv('out/{}/{}-meta.tsv'.format(set_type, file_name), sep='\t')
    labels = labels['zawod'].to_numpy()
    return data, labels

def select_sent(sentences):
    for sentence in sentences:
        cleared_sentence, targets = clear_sentence_and_locate_entities(sentence)
        if len(targets) > 0 and targets[0]["entity"] =='Stanisław Szymecki' :
            sent = ' '.join(cleared_sentence)
    return sent

def vectorize(sents):
    embeddings = createEmbeddings('flair')
    assert(isinstance(sents, list))
    emb_list = [] 
    for sent in sents:
        print(sent, '\n')
        embeddings_of_tokens = get_embedding(sent, embeddings)
        ans = np.mean([emb.numpy() for emb in embeddings_of_tokens], axis = 0)
        if isinstance(ans, (np.ndarray)):
            emb_list.append(ans)
        else:
            emb_list.append(np.zeros((1, 2048))[0])
    return emb_list


from sklearn.base import BaseEstimator, TransformerMixin

class Vectorizor(BaseEstimator, TransformerMixin):
    #Class Constructor 
    def __init__(self):
        pass
    
    #Return self nothing else to do here    
    def fit(self, X):
        return self 
    
    #Method that describes what we need this transformer to do
    def transform(self, X):
        return vectorize(X)


docs = load_files(env.learning_data_path, 'train')
people = list_people(docs)
categories = np.unique([p['category'] for p in people])



file  = 'categorization/doc3111'
with open(file, encoding='utf-8') as f:
    text = f.read().splitlines()

tokenizer = nltk.data.load('tokenizers/punkt/polish.pickle')
sentences = extract_sentences(text[0], tokenizer)
sent = select_sent(sentences)

mlp = MLPClassifier(hidden_layer_sizes=(256,128,64), activation='relu', solver='adam', max_iter=10, verbose=True)
x_train, y_train = load_data('train', 'single-5')
x_test, y_test = load_data('test', 'single-5')
x_train, y_train, encoder = preprocess_data(x_train, y_train)
ind = np.where(y_test == 'politycy')[0]
x_test = np.delete(x_test, ind, axis=0)
y_test = np.delete(y_test, ind, axis=0)
y_test = encoder.transform(y_test)

mlp.fit(x_train, y_train)   

from sklearn.pipeline import make_pipeline

vectorizer = Vectorizor()
ksiadz = "Mszy św. w kościele św. Apostołów Piotra i Pawła przewodniczy abp Edward Ozorowski, metropolita białostocki, który także wygłosi homilię."
ksiadz,_ = clear_sentence_and_locate_entities(ksiadz)
polityk = 'W poniedziałek minister sprawiedliwości <Entity name="Marek Biernacki" type="person" category="politycy">Marek Biernacki</Entity> poinformował, że służba więzienna znalazła w celi Mariusza T. materiały, które wiążą się z przestępstwami'
polityk,_ = clear_sentence_and_locate_entities(polityk)
dziennikarz = 'Antoni Łepkowski przekonywał wcześniej, że naczelny "GW" <Entity name="Adam Michnik" type="person" category="dziennikarze">Adam Michnik</Entity>, "kryjąc się za plecami Agory" i żądając 50 tys. zł zadośćuczynienia, chce w ten sposób tamować debatę publiczną.'
dziennikarz,_ = clear_sentence_and_locate_entities(dziennikarz)
aktor = ' Większość widzów, mimo starań samego <Entity name="Maciej Stuhr" type="person" category="aktorzy">Stuhra</Entity>, nadal kojarzy go z kreacjami komediowymi'
aktor,_ = clear_sentence_and_locate_entities(aktor)

c = make_pipeline(vectorizer, mlp)
c.predict_proba([aktor]).round(2)

explainer = LimeTextExplainer(class_names=encoder.inverse_transform(range(9)))
exp = explainer.explain_instance(aktor, c.predict_proba, num_samples=300, labels=[0,8])

print ('Explanation for class %s' % categories[0])
print ('\n'.join(map(str, exp.as_list(label=8))))

plt.bar(*zip(*exp.as_list(label=8)))
plt.show()