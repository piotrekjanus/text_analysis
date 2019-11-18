from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import glob 
import string
import re
# import morfeusz2
import nltk
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings, BertEmbeddings
from tqdm import tqdm
import numpy as np
import pickle
import env


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

def remove_entity(sentence):
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

## files needed for tensorflow projector
def save_model(entities_and_embeddings):
    with open('word2vec_emb.tsv','w', encoding='utf-8') as vec_file, open('word2vec_meta.tsv','w', encoding='utf-8') as metafile:
        for entity, embeddings in entities_and_embeddings.items():
            for embedding in embeddings:
                vec = [v.numpy() for v in embedding]
                vec = np.mean(vec, axis=0)
                vec = '\t'.join(map(str, vec))
                vec_file.write(vec+'\n')
                metafile.write(entity+'\n')

def clear_sentence(sentence):
    ## Remove digits
    sentence = sentence.translate(str.maketrans('', '', string.digits))

    ## Remove punctuation marks
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    return sentence

def clear_sentence_and_locate_entities(sentence, stopwords=STOPWORDS):
    entities_and_used_form = re.findall('<Entity name="(.*?)".*?">(.*?)</', sentence)

    ## Remove entity
    ## <Entity name="Tomasz Sekielski" type="person" category="dziennikarze">Tomasz Sekielski</Entity>
    ## will result as Tomasz Sekielski
    sentence = remove_entity(sentence)

    sentence = clear_sentence(sentence)

    ## Remove stopwords
    cleared_sentence = [word for word in sentence.split() if word not in stopwords and len(word) > 1]
    # print(cleared_sentence)
    sentence = cleared_sentence
    targets = []
    already_parsed = 0
    for entity, used_form in entities_and_used_form:
        used_form = clear_sentence(used_form)
        used_form_splitted = used_form.split()
        start = sentence[already_parsed:].index(used_form_splitted[0]) + already_parsed
        stop = start + len(used_form_splitted)
        already_parsed = stop
        targets.append({'start': start, 'length': len(used_form_splitted), 'used_form': used_form, 'entity': entity})
    # print(targets)

    cleared_sentence = [word.lower() for word in cleared_sentence]

    return cleared_sentence, targets

def get_embedding(sentence, embeddings):
    sentence = Sentence(sentence)
    embeddings.embed(sentence)
    return [token.embedding for token in sentence]

def createEmbeddings(name):
    if name == 'bert':
        return BertEmbeddings('bert-base-multilingual-cased')
    if name == 'flair':
        return FlairEmbeddings('polish-forward')

def get_embeddings_of_entity_in_corpus(documents, window_size = 5, method = 'bert'):
    # Polish word embeddings
    output = {}
    embeddings = createEmbeddings(method)

    for document_id, document in enumerate(documents):
        for sentence in tqdm(document):
            try:
                cleared_sentence, targets = clear_sentence_and_locate_entities(sentence)
                if len(targets) == 0:
                    continue
                embeddings_of_tokens = get_embedding(' '.join(cleared_sentence), embeddings)
                assert (len(embeddings_of_tokens) == len(cleared_sentence))
                for target in targets:
                    neighboring_embeddings = [embeddings_of_tokens[target['start'] - window_size: target['start']]] + \
                                             [embeddings_of_tokens[target['start'] + target['length']: target['start'] + target['length'] + window_size]]

                    if target['entity'] not in list(output.keys()):
                        output[target['entity']] = {document_id: [neighboring_embeddings]}
                    elif document_id not in output[target['entity']].keys():
                        output[target['entity']][document_id] = [neighboring_embeddings]
                    else:
                        output[target['entity']][document_id].append(neighboring_embeddings)

            except:
                print(f"failed to process: {sentence}")
    return output

def save_embeddings(embeddings):
    with open(env.outPath + '/embeddings.pickle', 'wb+') as f:
        pickle.dump(embeddings, f)


def read_files(method, test = False):
    docs = load_files(env.learningDataPath)

    ## list all people marked in text
    ## returns list of dicts, each person has attr: name, category, type 
    people = list_people(docs)

    ## returns list of sentences in each document
    tokenizer = nltk.data.load('tokenizers/punkt/polish.pickle')
    if test:
        docs = docs[0:50]
    documents = [extract_sentences(document, tokenizer) for document in docs]

    embeddings = get_embeddings_of_entity_in_corpus(documents, 5, method)
    save_embeddings(embeddings)


if __name__ == "__main__":
    read_files('bert')
