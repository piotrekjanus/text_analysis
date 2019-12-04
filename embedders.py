
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings, BertEmbeddings

import tensorflow_hub as hub
import numpy as np
import tensorflow_text

class WordEmbedder:

    def __init__(self, algorithm, sentence):
        self.model = self.get_model(algorithm)
        self.embeddings = self.get_embedding(''.join(sentence))
        assert len(self.embeddings) == len(sentence)

    def get_model(self, name):
        if name == 'bert':
            return BertEmbeddings('bert-base-multilingual-cased')
        if name == 'flair':
            return FlairEmbeddings('polish-forward')

    def get_embedding(self, sentences):
        sentence = Sentence(sentence)
        self.model.embed(sentence)
        return [token.embedding for token in sentence]

    def get(self, start, stop):
        return self.embeddings[start : stop]


class SentenceEmbedder:

    def __init__(self, sentence):
        self.model = self.get_model()
        self.sentence = sentence
    
    def get_model(self):
        return hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/2")

    def get(self, start, stop):
        model_input = ''.join(self.sentence[start:stop])
        return self.model([model_input])['outputs'].numpy()


if __name__ == '__main__':

    # test muse
    sentence = 'Ala ma kota'.split()
    embedder = SentenceEmbedder(sentence)
    print( embedder.get(0, 3) )

