from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import glob 
import string
import re
# import morfeusz2
import nltk
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings

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

def clear_sentence_and_locate_entities(sentence, stopwords=STOPWORDS):
    entities_and_used_form = re.findall('<Entity name="(.*?)".*?">(.*?)</', sentence)

    ## Remove entity
    ## <Entity name="Tomasz Sekielski" type="person" category="dziennikarze">Tomasz Sekielski</Entity>
    ## will result as Tomasz Sekielski
    sentence = remove_entity(sentence)

    ## Remove digits
    sentence = sentence.translate(str.maketrans('', '', string.digits))

    ## Remove punctuation marks
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))

    ## Remove stopwords
    cleared_sentence = [word for word in sentence.split() if word not in stopwords and len(word) > 1]
    print(cleared_sentence)
    sentence = cleared_sentence
    targets = []
    already_parsed = 0
    for entity, used_form in entities_and_used_form:
        used_form_splitted = used_form.split()
        start = sentence[already_parsed:].index(used_form_splitted[0]) + already_parsed
        stop = start + len(used_form_splitted)
        already_parsed = stop
        targets.append({'start': start, 'length': len(used_form_splitted), 'used_form': used_form, 'entity': entity})
    print(targets)

    cleared_sentence = [word.lower() for word in cleared_sentence]

    return cleared_sentence, targets

def get_flair_embedding(sentence, flair_embeddings):
    flair_sentence = Sentence(sentence)
    flair_embeddings.embed(flair_sentence)
    return [token.embedding for token in flair_sentence]

def get_embeddings_of_entity_in_sequences(sentences, entity, window_size):
    # Polish word embeddings
    polish_flair_embeddings = FlairEmbeddings('polish-forward')
    for sentence in sentences:
        cleared_sentence, targets = clear_sentence_and_locate_entities(sentence)
        embeddings_of_tokens = get_flair_embedding(' '.join(cleared_sentence), polish_flair_embeddings)
        assert (len(embeddings_of_tokens) == len(cleared_sentence))
        for target in [ target for target in targets if target['entity'] == entity]:
            neighboring_embeddings = embeddings_of_tokens[target['start'] - window_size: target['start']] + \
                                     embeddings_of_tokens[target['start'] + target['length']: target['start'] + target['length'] + window_size]
            neighbourhood = cleared_sentence[target['start'] - window_size: target['start']] + \
                            cleared_sentence[target['start'] + target['length']: target['start'] + target['length'] + window_size]

            print(neighboring_embeddings)
            print(neighbourhood)



if __name__ == "__main__":
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
    
    test_sent = 'cośtam coś tam <Entity name="Tomasz Sekielski" type="person" category="dziennikarze">Tomasza Sekielskiego</Entity> Llalal '
    test_sent += test_sent
    test_sent += 'cośtam coś tam <Entity name="Ktoś inny" type="person" category="dziennikarze">Ktoś inny</Entity> lalalal '

    print(test_sent)
    get_embeddings_of_entity_in_sequences([test_sent], 'Tomasz Sekielski', 1)
