import os
import numpy as np
from natural_language_processing.sentiment_data_processing import TextProcessing

"""In contrast to TextProcessing class which holds external data
ParseWordEmbeddings is a pretrained  network. A saved network that was previously
trained on a large dataset.If this original dataset is large enough and general enough
the hierarchy of features learned by the pretrained network can effectively act as 
a generic model.But for very specific tasks word embeddings make things worse.
It is flexible to changes for fine tunning unfreezing a few of
the top layers of a frozen model, which hold the more abstract patterns in order 
to make them more relevant for the problem at hand, to improve the accuracy around 1%.
But we avoid to spend so much time, so we do not change the class, so we make use of
classmethod decorator.
"""

class ParseWordEmbeddings:

    from natural_language_processing.configurations.configuration_infrastructure import Config
    from natural_language_processing.configurations.configurations import CFG
    config = Config.from_json(CFG)
    glove_dir = config.external_data_sources.word_embeddings
    file_name = config.external_data_sources.embeddings_file_name

    # embeddings have coefficients of similarity creating he word vectors
    @classmethod
    def create_embeddings_index(cls):
        embeddings_index = {}
        with open(os.path.join(cls.glove_dir, cls.file_name)) as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                # list consisting of a word and all the weights (coefs)  computed from the NN
                embeddings_index[word] = coefs
        print("values len; ", len(values))
        print("word len; ", len(word))
        print("embeddings_index len:", len(embeddings_index))

        print('Found %s coefficients.' % len(coefs))
        print('Found %s word vectors.' % len(embeddings_index))
        print("Please wait.The embeddings take time.")
        return embeddings_index

    def parse_embeddings(self):
        max_words = self.config.data.max_words
        embedding_matrix = np.zeros((max_words,
                                      self.config.external_data_sources.embeddings_dimension))

        data_pr = TextProcessing()
        word_index = data_pr.get_word_index()

        print("word_index: ", word_index)
        for word, i in word_index.items():
            if i< max_words:
                embedding_vector = ParseWordEmbeddings.create_embeddings_index().get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
        print("embedding_matrix : ", embedding_matrix)



data_process = TextProcessing()
data_process.process_train_data()

parser = ParseWordEmbeddings.create_embeddings_index()
# parser.parse_embeddings()
