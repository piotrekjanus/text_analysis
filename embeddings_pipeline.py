from enum import Enum
import os
import env

from elmo import elmo_read_and_generate_vecs, save_one_mention_context, save_document_context, save_corpus_context
from read_files import read_files
from process_embeddings import generate_embeddings

class Method(Enum):
    ELMO = 1
    FLAIR = 2
    BERT = 3

class Context(Enum):
    ONE_MENTION = 1
    DOCUMENT = 2
    CORPUS = 3

def context2string(context: Context):
    if context == Context.ONE_MENTION:
        return 'single'
    if context == Context.DOCUMENT:
        return 'document'
    if context == Context.CORPUS:
        return 'corpus'


def create_missing_dirs(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)


def embeddings_pipeline(corpus_dir: str, out_dir: str, method: Method, context: Context, window):
    """
    corpus_dir: path to folder containing documents with articles
    out_dir: path to folder where the function will store embeddings and metadata
    method: type of embedding
    context: context
    window: number of words before and after person to be calculated
    """

    create_missing_dirs([out_dir, env.tmp_data_path])

    if method == Method.ELMO:
        [vecs, corpus_list, person_list, document_dict, person_dict, profession_dict] = elmo_read_and_generate_vecs(corpus_dir, window)

        if context == Context.ONE_MENTION:
            output_name_prefix = f'{out_dir}/single-{window}'
            save_one_mention_context(vecs, person_list, profession_dict, output_name_prefix)
        if context == Context.DOCUMENT:
            output_name_prefix = f'{out_dir}/document-{window}'
            save_document_context(vecs, person_list, document_dict, person_dict, profession_dict, output_name_prefix)
        if context == Context.CORPUS:
            output_name_prefix = f'{out_dir}/corpus-{window}'
            save_corpus_context(vecs, person_list, person_dict, profession_dict, output_name_prefix)

    if method == Method.BERT:
        read_files(corpus_dir, 'bert')

    if method == Method.FLAIR:
        read_files(corpus_dir, 'flair')

    if method == Method.BERT or method == Method.FLAIR:
        generate_embeddings(corpus_dir, out_dir, context2string(context), window)


embeddings_pipeline('../../test-data', 'data/test', Method.FLAIR, Context.ONE_MENTION, 5)
