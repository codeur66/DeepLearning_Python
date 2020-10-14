
from natural_language_processing.configurations.configuration_infrastructure import Config
from natural_language_processing.configurations.configurations import CFG

"""TextProcessing, different data sources
may be added in the pipeline for other models or the current may need extension.
In contrast to ParseWordEmbeddings class which holds external data source and pretrained.
"""

from keras.preprocessing.text import Tokenizer


class TextProcessing:

    def __init__(self):
        self.config = Config.from_json(CFG)
        self.texts, self.labels = [], []
        self.max_len = self.config.data.maxlen
        self.training_samples = self.config.data.training_samples
        self.validation_samples = self.config.data.validation_samples
        self.max_words = self.config.data.max_words
        self.tokenizer = Tokenizer(num_words=self.max_words)

    # Reads the folders stores in data structures the corresponding data (data and labels) for the train.
    def process_train_data(self):
        import os
        train_dir = os.path.join(self.config.data.path_train_data, 'train')
        for label_type in ['neg', 'pos']:
            dir_name = os.path.join(train_dir, label_type)
            for file_type in os.listdir(dir_name):
                if file_type[-4:] == '.txt':
                    fl_path = os.path.join(dir_name, file_type)
                    f = open(fl_path)  # opens only the .txt files
                    self.texts.append(f.read()) # The text of each .txt appends to a universal structure
                    f.close()
                    if label_type == 'neg': # The labels of each .txt appends to a universal structure
                        self.labels.append(0)
                    else:
                        self.labels.append(1)

    # Counts the words with an incremental index and informs the dictionary of tokenizer
    # <word , index_number>
    def word_counter_informs_tokenizer(self):
        self.tokenizer.fit_on_texts(self.texts)
        word_index = self.tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        return self  # chainning with method tokenize_and_store_data, they go together


    def tokenize_and_store_data(self):
        from keras.preprocessing.sequence import pad_sequences  # the lists into 2D of same sizes.
        import numpy as np
        # convert text to numbers, most frequent words will be taken into account
        sequences = self.tokenizer.texts_to_sequences(self.texts)
        # each list of words here represented with unique sequential numbers
        # must have the same size, zero paddings does this.
        data = pad_sequences(sequences, maxlen=self.max_len)
        labels = np.asarray(self.labels)
        print('Shape of data tensor:', data.shape)
        print('Shape of label tensor:', labels.shape)
        # creation of a incremental list of unique numbers for the sake of suffling the words.
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        # the words suffled randomly - we want fairness, now the structure is ready to store them
        data = data[indices]
        labels = labels[indices]
        return data, labels

    def split_data(self):
        x, y = self.tokenize_and_store_data()
        x_train = x[:self.training_samples]
        y_train = y[:self.training_samples]
        x_val = x[self.training_samples: self.training_samples + self.validation_samples]
        y_val = y[self.training_samples: self.training_samples + self.validation_samples]
        print("x_train, y_train, x_val, y_val ; ", x_train, y_train, x_val, y_val)
        return x_train, y_train, x_val, y_val




def __main__():
    data_proc = TextProcessing()
    data_proc.process_train_data()
    data_proc.word_counter_informs_tokenizer().tokenize_and_store_data()
    data_proc.split_data()