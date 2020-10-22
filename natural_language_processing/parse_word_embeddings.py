import os
import numpy as np
"""In contrast to TextProcessing class which holds external data
ParseWordEmbeddings is a pretrained  network. A saved network that was previously
trained on a large dataset.If this original dataset is large enough and general enough
the hierarchy of features learned by the pretrained network can effectively act as 
a generic model.But for very specific tasks word embeddings make things worse.
It is flexible to changes for fine tunning unfreezing a few of
the top layers of a frozen model, which hold the more abstract patterns in order 
to make them more relevant for the problem at hand, to improve the accuracy around 1%.
But we avoid to spend so much time, so we will not change this class, so we make use of
classmethod decorator.
"""

from natural_language_processing.configurations.configuration_infrastructure import Config
from natural_language_processing.configurations.configurations import CFG

class ParseWordEmbeddings:

    config = Config.from_json(CFG)
    glove_dir = config.external_data_sources.word_embeddings
    file_name = config.external_data_sources.embeddings_file_name

    # embeddings is a dictionary that maps the word indices to an associated vector.
    @classmethod
    def embeddings_vectors(cls):
        embedding_indexed_vectors = {}
        with open(os.path.join(cls.glove_dir, cls.file_name)) as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                # a list consists a word and all the weights (coefs) computed from the NN
                embedding_indexed_vectors[word] = coefs
        print('Found %s coefficients.' % len(coefs))
        print('Found %s word vectors.' % len(embedding_indexed_vectors))
        return embedding_indexed_vectors

    # All sequences in a batch must have the same length to pack them into a single tensor,
        # so we do zero padding to shorter sequences, and the longer sequences are truncated.

    @classmethod
    def create_embeddings_matrix(cls, word_index):
        max_words = cls.config.data.max_words
        embedding_matrix = np.zeros((max_words,
                                     cls.config.external_data_sources.embeddings_dimension))
        embedding_indexed_vectors = cls.embeddings_vectors()
        for word, i in word_index.items():
            if i < max_words:
                embedding_vector = embedding_indexed_vectors.get(word)
                if embedding_vector is not None:  # words not found in the embedding index will be zeros
                    embedding_matrix[i] = embedding_vector
        print("embedding_matrix : ", embedding_matrix)
        return embedding_matrix



