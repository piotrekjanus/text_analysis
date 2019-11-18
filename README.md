# Text analysis

This is a simple pipeline for parsing and creating embeddings of "Entity Corpus". 

# Local variables
Before running the code, create file `env.py` in main directory (ignored by Git), that contains the following variables:

``` Python
learning_data_path = '[path]'
tmp_data_path = '[path]'
elmo_pl_path = '[path]'
```

# Downloading ELMO
Before running the code download and extract ELMO pretreined for Polish to directory indicated in `env.py` from https://drive.google.com/file/d/110c2H7_fsBvVmGJy08FEkkyRiMOhInBP/view?usp=sharing


# Generating any embedding (ELMO/FLAIR/BERT)
In order to generate embedding using selected method, context and neighbourhood, use function `embeddings_pipeline` from `embeddings_pipeline.py`.
It is an "interface" to implementations of those 3 methods.


# Description of read_files.py and process_embeddings.py

Pipeline is splited into two parts:
* text processing
* creation of entity embeddings

## text processing
* loading corpus
* preprocesses it (removing interpunction, and stopwords etc.)
* calculates embedding of chosen form for each word
* locates entities

## creation of emtity embeddings
* generates embeddings with specified context level, and window size
