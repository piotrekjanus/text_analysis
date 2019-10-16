FILE_NAME = 'embeddings.pickle'

def save_model(person_vec):
    with open('word2vec_emb.tsv','w', encoding='utf-8') as vec_file, open('word2vec_meta.tsv','w', encoding='utf-8') as metafile:
        for entity, embeddings in person_vec:
            vec = '\t'.join(map(str, embeddings))
            vec_file.write(vec+'\n')
            metafile.write(entity+'\n')

def prepare_vector(embed, context='single'):
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
                        vec = [v.numpy() for v in sent]
                        vec = np.mean(vec, axis=0) if len(vec)>1 else vec
                    else:
                        vec = sent.numpy() 
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
                        vec = [v.numpy() for v in sent]
                        vec = np.mean(vec, axis=0) if len(vec)>1 else vec
                    else:
                        vec = sent.numpy() 
                    if len(vec)>0:
                        doc_sent.append(vec)
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
                        vec = [v.numpy() for v in sent]
                        vec = np.mean(vec, axis=0) if len(vec)>1 else vec
                    else:
                        vec = sent.numpy() 
                    if len(vec)>0:
                        corp_sent.append(vec)
            vec = np.mean(corp_sent, axis=0)    
            person_vec.append((person, vec))
    save_model(person_vec)

if __name__ == "__main__":
    with open(FILE_NAME, 'rb') as f:
        embed = pickle.load(f)
