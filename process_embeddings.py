FILE_NAME = 'embeddings.pickle'
import pickle
import numpy as np

def save_model(person_vec):
    with open('word2vec_emb.tsv','w', encoding='utf-8') as vec_file, open('word2vec_meta.tsv','w', encoding='utf-8') as metafile:
        for entity, embeddings in person_vec:
            vec = '\t'.join(map(str, embeddings))
            vec_file.write(vec+'\n')
            metafile.write(entity+'\n')

def prepare_vector(embed, context='single', window=5):
    person_vec = []

    if context=='single':
        for person in embed.keys():
            ## Get all documents
            docs = list(embed[person].values())
            ## For doc in docs 
            for doc in docs:
                ## For sent in doc
                for sent in doc:
                    if isinstance(sent, list):
                        before = sent[0]
                        after = sent[1]
                        vec = before[:window] + after[:window]
                        vec = [v.numpy() for v in vec]
                        if len(vec)<2:
                            vec = vec[0] if len(vec)==1 else vec
                        else:
                            vec = np.mean(vec, axis=0) 
                    else:
                        vec = sent.numpy() 
                    
                    if isinstance(vec, (np.ndarray, np.generic)):
                        person_vec.append((person, vec))

    if context=='document':
        for person in embed.keys():
            ## Get all documents
            docs = list(embed[person].values())
            ## For doc in docs 
            for doc in docs:
                ## For sent in doc
                doc_sent = []
                for sent in doc:
                    if isinstance(sent, list):
                        before = sent[0]
                        after = sent[1]
                        vec = before[:window] + after[:window]
                        vec = [v.numpy() for v in vec]
                        if len(vec)<2:
                            vec = vec[0] if len(vec)==1 else vec
                        else:
                            vec = np.mean(vec, axis=0) 
                    else:
                        vec = sent.numpy() 
                    if isinstance(vec, (np.ndarray, np.generic)):
                        doc_sent.append(vec)
                if len(doc_sent)>0:
                    vec = np.mean(doc_sent, axis=0)
                    person_vec.append((person, vec))
                
    if context=='corpus':
        for person in embed.keys():
            ## Get all documents
            docs = list(embed[person].values())
            corp_sent = []
            ## For doc in docs 
            for doc in docs:
                ## For sent in doc
                for sent in doc:
                    if isinstance(sent, list):
                        before = sent[0]
                        after = sent[1]
                        vec = before[:window] + after[:window]
                        vec = [v.numpy() for v in vec]
                        if len(vec)<2:
                            vec = vec[0] if len(vec)==1 else vec
                        else:
                            vec = np.mean(vec, axis=0)
                    else:
                        vec = sent.numpy() 
                    if len(vec)>0:
                        corp_sent.append(vec)
            vec = np.mean(corp_sent, axis=0)    
            person_vec.append((person, vec))
    return person_vec

if __name__ == "__main__":
    with open(FILE_NAME, 'rb') as f:
        embed = pickle.load(f)
    to_save = prepare_vector(embed, context='single', window=3)
    for a,b in to_save:
        print(len(b))
    save_model(to_save)


