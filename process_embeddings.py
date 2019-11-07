FILE_NAME = 'embeddings.pickle'
import pickle
import numpy as np

def save_model(person_vec, file_name):
    with open(f'{file_name}-emb.tsv','w', encoding='utf-8') as vec_file, open(f'{file_name}-meta.tsv','w', encoding='utf-8') as metafile:
        metafile.write('Imie\ttyp\tzawod'+'\n')
        for entity, embeddings in person_vec:
            try:
                vec = '\t'.join(map(str, embeddings))
                entity_vec = [list(k.values()) for k in people if k['name'] == entity]
                if len(entity)>0:
                    entity_vec = entity_vec[0]
                    entity_vec = '\t'.join(map(str, entity_vec))
                else:
                    entity_vec = entity+'\t \t '
                vec_file.write(vec+'\n')
                metafile.write(entity_vec+'\n')
            except:
                pass

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
            if len(corp_sent) > 0:
                vec = np.mean(corp_sent, axis=0)
                person_vec.append((person, vec))
    return person_vec

if __name__ == "__main__":
    from tqdm import tqdm
    with open(FILE_NAME, 'rb') as f:
        embed = pickle.load(f)

    for context in ['single','document','corpus']:
        for window in tqdm([1,2,3,4,5]):
            to_save = prepare_vector(embed, context=context, window=window)
            save_model(to_save, f'{context}-{window}')