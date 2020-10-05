import os
import numpy as np

# In contrast to SentimentDataProcessing this class which holds external..
# data source, pretrained not flexible to change or expansion.


class ParseWordEmbeddings:

    from natural_language_processing.configurations.configuration_infrastructure import Config
    from natural_language_processing.configurations.configurations import CFG
    config = Config.from_json(CFG)
    glove_dir = config.external_data_sources.word_embeddings
    file_name = config.external_data_sources.embeddings_file_name

    @classmethod
    def parse(cls):
        embeddings_index = {}
        with open(os.path.join(cls.glove_dir, cls.file_name)) as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        print('Found %s coefficients.' % len(coefs))
        print('Found %s word vectors.' % len(embeddings_index))
        #print('word : %s , coefs : %s' % (word, coefs))
        print("Please wait.The embeddings take time.")
        print("embeddings_index : ", embeddings_index)
        return embeddings_index


ParseWordEmbeddings().parse()
