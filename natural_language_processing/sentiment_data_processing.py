
from natural_language_processing.configurations.configuration_infrastructure import Config
from natural_language_processing.configurations.configurations import CFG

"""SentimentDataProcessing become a class for 
the reason of many internal data sources
may be added in the pipeline or the current may change.
In contrast to ParseWordEmbeddings class which holds external data source and pretrained.
"""

class SentimentDataProcessing():

    def __init__(self):
        self.config = Config.from_json(CFG)
        self.texts, self.labels = self.process_train_data()
        self.maxlen = self.config.data.maxlen
        self.training_samples = self.config.data.training_samples
        self.validation_samples = self.config.data.validation_samples
        self.max_words = self.config.data.max_words

    # Reads the folders stores in data structures the corresponding data (data and labels) for the train.
    def process_train_data(self):
        import os
        train_dir = os.path.join(self.config.data.path_train_data, 'train')
        print("train_dir : ", train_dir)
        self.texts = []
        self.labels = []
        for label_type in ['neg', 'pos']:
            dir_name = os.path.join(train_dir, label_type)
            for file_type in os.listdir(dir_name):
                if file_type[-4:] == '.txt':
                    fl_path = os.path.join(dir_name, file_type)
                    f = open(fl_path)  # opens only the .txt files
                    self.texts.append(f.read())
                    f.close()
                    if label_type == 'neg':
                        self.labels.append(0)
                    else:
                        self.labels.append(1)
        return self.texts, self.labels

    def tokenize_train_data(self):
        from keras.preprocessing.text import Tokenizer
        from keras.preprocessing.sequence import pad_sequences  # to make the lists insto 2D of same sizes.
        import numpy as np
        tokenizer = Tokenizer(num_words=self.max_words)
        tokenizer.fit_on_texts(self.texts)
        sequences = tokenizer.texts_to_sequences(self.texts)
        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        data = pad_sequences(sequences, maxlen=self.maxlen)
        labels = np.asarray(self.labels)
        print('Shape of data tensor:', data.shape)
        print('Shape of label tensor:', labels.shape)
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]
        return data, labels

    def split_train_validation(self):
        x_train = self.tokenize_train_data()[0][:self.training_samples]
        y_train =  self.tokenize_train_data()[1][:self.training_samples]
        x_val = self.tokenize_train_data()[0][self.training_samples: self.training_samples + self.validation_samples]
        y_val = self.tokenize_train_data()[1][self.training_samples: self.training_samples + self.validation_samples]
        print("x_train, y_train, x_val, y_val", x_train.shape, y_train.shape, x_val.shape, y_val.shape)
        return x_train, y_train, x_val, y_val


data_proc = SentimentDataProcessing()
data_proc.process_train_data()
data_proc.tokenize_train_data()
split_data = data_proc.split_train_validation()