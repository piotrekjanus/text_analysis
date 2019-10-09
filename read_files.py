from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import glob 
import string
import re
import morfeusz2
PATH = 'categorization/learningData'


with open('stopwords.txt', encoding='utf-8') as f:
    STOPWORDS = f.read().splitlines()

def load_files(path):
    docs = []
    files = glob.glob(path+'/**/*', recursive=True)
    for file in files:
        try:
            with open(file, encoding='utf-8') as f:
                docs.append(f.read().splitlines())
        except:
            pass
    return docs

def wyjeb_to_entity(sentence):
    return re.sub('\<(.*?)\>','', sentence) 

def extract_lemm(morf):
    return morf[0][2][1].split(':', 1)[0]

def text_preprocess(files, stopwords):
    docs = []
    ## http://morfeusz.sgjp.pl/download/
    morf = morfeusz2.Morfeusz()
    for file in files:
        sentence = file[0]
        ## Usuń entity
        ## np. <Entity name="Tomasz Sekielski" type="person" category="dziennikarze">Tomasz Sekielski</Entity>
        sentence = wyjeb_to_entity(sentence)

        ## Usuń cyferki
        sentence = sentence.translate(str.maketrans('', '', string.digits))

        ## Usuń znaki interpunkcyjne
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))

        ## Usuń stopwords
        sentence = [word for word in sentence.split() if word not in stopwords and len(word)>1]

        ## Lematyzacja
        ## wyglada jak gówno ale działa
        ## nie wiem czy jest funkcja, ktora zwraca sam lemat, wiec trzeba robic takie czary.
        ## Czasami zwraca różne wyniki np. dla slowa zamek zwraca zamek:s1, zamek:s2
        ## Try: morf.analyse('zamki')
        ##      morf.analyse('zamki')[0][2][1].split(':', 1)[0]
        sentence = [extract_lemm(morf.analyse(word)) for word in sentence]

        ## zrup maue
        sentence = [word.lower() for word in sentence]

        docs.append(sentence)
    return docs 

def make_word2vec(sentences):
    model = Word2Vec(sentences, size=100, window=5, min_count=3, workers=12)
    model.train(sentences, total_examples=len(sentences), epochs=5)
    return model

## files needed for tensorflow projector
def save_model(model):
    with open('word2vec_emb.tsv','w', encoding='utf-8') as vec_file, open('word2vec_meta.tsv','w', encoding='utf-8') as metafile:
        for word in list(model.wv.vocab):
            vec = '\t'.join(map(str, model[word]))
            vec_file.write(vec+'\n')
            metafile.write(word+'\n')

if __name__ == "__main__":
    ## można zrobić yield
    docs = load_files(PATH)
    print('data loaded')
    ## jest mocno nieoptymalnie
    text = text_preprocess(docs, STOPWORDS)
    print('text preprocessed')
    ## train word2vec
    model = make_word2vec(text)
    print('model trained')
    save_model(model)
