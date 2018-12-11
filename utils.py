import nltk
import pickle
import re
import numpy as np
from gensim.utils import tokenize

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': 'starspace_embedding.tsv',
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""

    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.

      Args:
        embeddings_path - path to the embeddings file.

      Returns:
        embeddings - dict mapping words to vectors;
        embeddings_dim - dimension of the vectors.
      """

    with open(embeddings_path, 'r') as f:
        lines = f.readlines()

    embeddings = {}
    embeddings_dims = 0
    for line in lines:
        line_split = line.split()
        word = line_split[0]
        embedding_vector = [float(embedding_value) for embedding_value in line_split[1:]]
        embeddings[word] = embedding_vector
        embeddings_dims = embedding_vector

    return embeddings, embeddings_dims


def question_to_vec(question, embeddings, dim):
    """Transforms a string to an embedding by averaging word embeddings."""

    vector_list = []
    for token in list(tokenize(question, deacc=True)):
        if token in embeddings:
            vector_list.append(embeddings[token])

    if len(vector_list) == 0:
        return np.zeros(dim)
    else:
        return np.mean(vector_list, axis=0)


def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
