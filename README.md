# Text analysis

This is a simple pipeline for parsing and creating embeddings of "Entity Corpus". 

Pipeline is splited into two parts:
* text processing
* creation of entity embeddings

## text processing
* loading corpus
* preprocesses it (removing interpunction, and stopwords etc.)
* calculates embedding of chosen form for each word
* locates entities

## creation pf emtity embeddings
* generates embeddings with specified context level, and window size
