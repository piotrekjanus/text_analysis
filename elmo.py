from overrides import overrides
from allennlp.modules.elmo import Elmo, batch_to_ids
import os, os.path
import re
import numpy as np
from tqdm import tqdm

import env
from elmo_helper import *

options_file = env.elmo_pl_path + '/options.json'
weight_file = env.elmo_pl_path + '/weights.hdf5'

def read_corpus(path, size, test = False):
    """
    INPUT:
    path - ścieżka do folderu z dokumentami (korpus)
    size - rozmiar sąsiedztwa budującego pojedynczą frazę (liczba słów w tył i w przód względem nazwiska)
    
    OUTPUT:
    corpus_list - lista fraz (fraza z kolei jest listą wyrazów składających się na nią)
    person_list - lista osób (imiona i nazwiska) odpowiadających poszczególnych frazom z 'corpus_list'
    document_dict - słownik, którego kluczami są numery dokumentów, a wartościami - zbiory indeksów,
                    na których w 'corpus_list' występują frazy z tego dokumentu
    person_dict - słownik, którego kluczami są imiona i nazwiska osób, a wartościami - zbiory indeksów,
                    na których w 'corpus_list' występują frazy dotyczące tych osób
    profession_dict - słownik człowiek - profesja
    """
    
    # file_number = int(max([ re.findall(r'\d+', name) for name in os.listdir(path)])[0])
    file_number = max([ int(re.findall(r'\d+', name)[0]) for name in os.listdir(path)])
    if test:
        file_number = 50
    people_vect_dict= {}
    corpus_list = []
    person_list = []
    profession_dict = {}

    document_dict = {}
    person_dict = {}
    for i in range(1,file_number):
        try:
            document_dict[i] = set()
            pathFile = path + r"/doc{}".format(i)
            l = stringify(pathFile)
            l= ''.join(c for c in l if c not in punctuation)
            l= ''.join(c for c in l if c not in words_with_dot)
            v = exclude_vectors_nsize(l,size)
            for s in v: # s - fraza - w sąsiedztwie
                if len(s[0]) > 0:
                    s_num = len(corpus_list)
                    corpus_list.append(s[0])
                    person_list.append(s[1])
                    if s[1] not in profession_dict:
                        profession_dict[s[1]] = s[2]
                    document_dict[i].add(s_num)
                    if s[1] not in person_dict:
                        person_dict[s[1]] = set()
                    person_dict[s[1]].add(s_num)
        except FileNotFoundError:
            continue
    
    return [corpus_list, person_list, document_dict, person_dict, profession_dict]


def save_one_mention_context(vecs, person_list, profession_dict, output_name):
    vectors = []
    labels = []
    
    for i in range(0, len(vecs)):
        vectors.append(vecs[i])
        labels.append(person_list[i])
    
    np.savetxt(output_name + "_om_vecs.tsv", np.array(vectors), delimiter="\t")
    with open(output_name + "_om_labels.tsv", 'w') as f:
        for label in labels:
            f.write(label + "\t" + profession_dict[label] + "\n")


def save_document_context(vecs, person_list, document_dict, person_dict, profession_dict, output_name):
    vectors = []
    labels = []
    
    for document in tqdm(document_dict):
        for person in person_dict:
            intersection = document_dict[document].intersection(person_dict[person])
            if len(intersection) > 0:
                document_person_vecs = [vecs[i] for i in intersection]
                vectors.append(np.average(document_person_vecs, axis=0))
                labels.append(person)
    
    np.savetxt(output_name + "_document_vecs.tsv", vectors, delimiter="\t")
    with open(output_name + "_document_labels.tsv", 'w') as f:
        for label in labels:
            f.write(label + "\t" + profession_dict[label] + "\n")


def save_corpus_context(vecs, person_list, person_dict, profession_dict, output_name):
    vectors = []
    labels = []
    
    for person in tqdm(person_dict):
        indexes = person_dict[person]
        if len(indexes) > 0:
            person_vecs = [vecs[i] for i in indexes]
            vectors.append(np.average(person_vecs, axis=0))
            labels.append(person)

    np.savetxt(output_name + "_corpus_vecs.tsv", vectors, delimiter="\t")
    with open(output_name + "_corpus_labels.tsv", 'w') as f:
        for label in labels:
            f.write(label + "\t" + profession_dict[label] + "\n")


def elmo_emb_2_vec(elmo_emb):

    np_emb = elmo_emb['elmo_representations'][0].detach().numpy()

    res = []

    for i in range(0, len(np_emb)):
        broke = False
        for j in range(0, len(np_emb[i])):
            if np.count_nonzero(np_emb[i][j]) == 0:
                res.append(np.average(np_emb[i][0:j], axis=0))
                broke = True
                break
        if not broke:
            res.append(np.average(np_emb[i], axis=0))
    
    return res


def elmo_read_and_generate_vecs(korpus: str, size : int, output_name: str, test = False):
    """
    INPUT:
    korpus - nazwa folderu z korpusem (np. "korpusGAZETA")
    size - rozmiar sąsiedztwa budującego pojedynczą frazę (liczba słów w tył i w przód względem nazwiska)
    output_name - przedrostek nazw plików wejściowych - bedą miały format "{output_name}_{oznaczenie_korpusu}_{labels/vecs}.tsv"
    
    OUTPUT:
    vecs - wektory przetworzone przez elmo - 1 wektor dla frazy zdubowanej napodstawie sąsiedztwa
    corpus_list, person_list, document_dict, person_dict - wynik działania funkcji 'read_corpus' - patrz komentarz do tamtej funkcji
    """
    
    elmo = Elmo(options_file, weight_file, 1, dropout=0)
    
    path_raw = f'{env.learning_data_path}/{korpus}'

    [corpus_list, person_list, document_dict, person_dict, profession_dict] = read_corpus(path_raw, size, test)
    
    vecs = []
    step = 1000
    
    for i in tqdm(range(0, len(corpus_list), step)):
        character_ids = batch_to_ids(corpus_list[i:i+step])

        embeddings = elmo(character_ids)

        vecs.extend(elmo_emb_2_vec(embeddings))
    
    return [vecs, corpus_list, person_list, document_dict, person_dict, profession_dict]


def generate_elmo_embeddings(korpus_name, window, test=False):
    save_name = f'{env.out_path}/{korpus_name}-{window}'
    [vecs, corpus_list, person_list, document_dict, person_dict, profession_dict] = elmo_read_and_generate_vecs(korpus_name, window, save_name, test)

    save_one_mention_context(vecs, person_list, profession_dict, save_name)
    save_document_context(vecs, person_list, document_dict, person_dict, profession_dict, save_name)
    save_corpus_context(vecs, person_list, person_dict, profession_dict, save_name)

if __name__ == "__main__":
    generate_elmo_embeddings('korpusGAZETA', 3, True)
