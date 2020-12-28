import os
import numpy as np

"""In contrast to TextProcessing class which holds external data
ParseWordEmbeddings is a pretrained  network. A saved network that was previously
trained on a large dataset.If this original dataset is large enough and general enough
the hierarchy of features learned by the pretrained network can effectively act as 
a generic model.But for very specific tasks word embeddings make things worse.
It is flexible to changes for fine tuning unfreezing a few of
the top layers of a frozen model, which hold the more abstract patterns in order 
to make them more relevant for the problem at hand, to improve the accuracy around 1%.
But we avoid to spend so much time, so we will not change this class, so we make use of
classmethod decorator.
"""

from natural_language_processing.configurations.configuration_infrastructure import Config
from natural_language_processing.configurations.configurations import CFG
from natural_language_processing.logging.LoggerCls import LoggerCls
import os.path


class ParseWordEmbeddings:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    formatter = '%(name)s - %(levelname)s  - %(message)s'
    logToFile = LoggerCls("log_to_file", "parse_word_ebeddings:", dir_path + "/data_piepeline.log", "w", formatter,
                        "INFO")
    logStream = LoggerCls("log_to_stdout", "WordEmbeddings: ", None, "w", formatter, "INFO")

    config = Config.from_json(CFG)
    glove_dir = config.external_data_sources.word_embeddings
    file_name = config.external_data_sources.embeddings_file_name

    # embeddings is a dictionary that maps the word indices to an associated vector.
    @classmethod
    def embeddings_vectors(cls):
        try:
            embedding_indexed_vectors = {}
            with open(os.path.join(cls.glove_dir, cls.file_name)) as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    # a list consists a word and all the weights (coefs) computed from the NN
                    embedding_indexed_vectors[word] = coefs
            ParseWordEmbeddings.logStream.info("Found %s coefficients." % len(coefs))
            ParseWordEmbeddings.logStream.info("Found %s word vectors." % len(embedding_indexed_vectors))

            # print('Found %s coefficients.' % len(coefs))
            # print('Found %s word vectors.' % len(embedding_indexed_vectors))
            return embedding_indexed_vectors
        except (IOError, FileNotFoundError, TypeError):
            ParseWordEmbeddings.logToFile.logger.("Error encountered on method <embeddings_vectors>")

    # All sequences in a batch must have the same length to pack them into a single tensor,
    # so we do zero padding to shorter sequences, and the longer sequences are truncated.

    @classmethod
    def create_embeddings_matrix(cls, word_index):
        try:
            max_words = cls.config.data.max_words
            embedding_matrix = np.zeros((max_words,
                                         cls.config.external_data_sources.embeddings_dimension))
            ParseWordEmbeddings.logStream.info("embedding_matrix shape : " .format(embedding_matrix.shape))
            # print("embedding_matrix shape : ", embedding_matrix.shape)
            embedding_indexed_vectors = cls.embeddings_vectors()
            for word, i in word_index.items():
                if i < max_words:
                    embedding_vector = embedding_indexed_vectors.get(word)
                    if embedding_vector is not None:  # if words not found in the embedding index will be zeros
                        embedding_matrix[i] = embedding_vector
            return embedding_matrix
        except BaseException("Error encountered on method <create_embeddings_matrix>"):
            ParseWordEmbeddings.logToFile.logger.error("Error encountered on method <create_embeddings_matrix>")

    @classmethod
    def store_h5py(cls, embedding_matrix):
        try:
            from h5py import File
            hdf = File(ParseWordEmbeddings.config.external_data_sources.HDFS_EXTERNAL_DATA_FILENAME, "w")
        except IOError:
            ParseWordEmbeddings.logToFile.logger.error("Error encountered on method <store h5py>. \
            The external file failed to open for write")
            # print("The external file failed to open for write.")
        else:
            hdf.create_dataset("external_dataset", data=embedding_matrix, compression="gzip")
            hdf.close()
