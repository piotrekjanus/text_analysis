from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import glob 
import string
import re
import morfeusz2
import nltk
PATH = 'categorization/learningData'


with open('stopwords.txt', encoding='utf-8') as f:
    STOPWORDS = f.read().splitlines()

def load_files(path):
    docs = []
    files = glob.glob(path+'/**/*', recursive=True)
    for file in files:
        try:
            with open(file, encoding='utf-8') as f:
                docs.extend(f.read().splitlines())
        except:
            pass
    return docs

def wyjeb_to_entity(sentence):
    return re.sub('\<(.*?)\>','', sentence) 

def extract_lemm(morf):
    return morf[0][2][1].split(':', 1)[0]

def list_people(docs):
    ## List all people appearing in documents 
    all_people = []
    added_people = []
    for doc in docs:
        entities = re.findall('\<Entity(.*?)\>', doc)
        for item in entities:
            name = re.findall('name="(.*?)"', item)[0]
            if name not in added_people:
                added_people.append(name)
                try:
                    person = {
                        'name': re.findall('name="(.*?)"', item)[0],
                        'type': re.findall('type="(.*?)"', item)[0],
                        'category': re.findall('category="(.*?)"', item)[0]
                    }
                    all_people.append(person)
                except:
                    pass
    return all_people

def extract_sentences(doc, tokenizer):
    ## You might need to execute
    ## nltk.download('punkt') before using nltk
    return(tokenizer.tokenize(doc))

def find_sent_with_person(p_name, all_sentences):
    selected_sentences = []
    for doc in all_sentences:
        for sentence in doc:
            if p_name in sentence:
                selected_sentences.append(sentence)
    return selected_sentences

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

def extract_neighbor_words(sentences, before=3, after=3, stopwords=STOPWORDS, keep_person=False):
    extracted_words= []
    ## http://morfeusz.sgjp.pl/download/
    morf = morfeusz2.Morfeusz()
    for sentence in sentences:
        ## Usuń entity
        ## np. <Entity name="Tomasz Sekielski" type="person" category="dziennikarze">Tomasz Sekielski</Entity>
        ## zostanie Tomasz Sekielski

        person = re.findall('">(.*?)</',sentence)[0].split()
        sentence = wyjeb_to_entity(sentence)

        ## Remove digits
        sentence = sentence.translate(str.maketrans('', '', string.digits))

        ## Remove punctuation marks
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        print(sentence)
        ## Remove stopwords
        sentence = [word for word in sentence.split() if word not in stopwords and len(word)>1]

        ## Find words before and after
        min_idx = min([sentence.index(word) for word in person])
        max_idx = max([sentence.index(word) for word in person])
        words_before_after = sentence[max(min_idx-before,0):min_idx] + sentence[max_idx+1:min(max_idx+after+1, len(sentence))]
        
        if keep_person:
            words_before_after = words_before_after + person

        ## Lematyzacja (opcjonalna wegług prowadzącej)
        ## nie wiem czy jest funkcja, ktora zwraca sam lemat, wiec trzeba robic takie czary.
        ## Czasami zwraca różne wyniki np. dla slowa zamek zwraca zamek:s1, zamek:s2
        ## Try: morf.analyse('zamki')
        ##      morf.analyse('zamki')[0][2][1].split(':', 1)[0]
        sentence = [extract_lemm(morf.analyse(word)) for word in words_before_after]

        ## chyba np. word2vec odróżnia wyrazy z wielkich i małych liter
        sentence = [word.lower() for word in sentence]
        extracted_words.append(sentence)

    return extracted_words

    

if __name__ == "__main__":
    ## można zrobić yield
    docs = load_files(PATH)

    ## list all people marked in text
    ## returns list of dicts, each person has attr: name, category, type 
    people = list_people(docs)

    ## returns list of sentences in each document
    tokenizer = nltk.data.load('tokenizers/punkt/polish.pickle')
    sentences = [extract_sentences(document, tokenizer) for document in docs]

    ## find sentences for specific person
    person1 = people[0]['name']
    filtered_sentences = find_sent_with_person(person1, sentences)
    
    a = extract_neighbor_words(filtered_sentences, before=1, after=1)
