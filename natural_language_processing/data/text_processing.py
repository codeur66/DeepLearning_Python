import numpy as np
from natural_language_processing.configurations.configuration_infrastructure import Config
from natural_language_processing.configurations.configurations import CFG
from keras.preprocessing.text import Tokenizer

"""TextProcessing, different data sources
may be added in the pipeline for other models or the current may need extension.
In contrast to ParseWordEmbeddings class which holds external data source - pretrained.
"""


class TextProcessing:
    from natural_language_processing.configurations.configuration_infrastructure import Config
    from natural_language_processing.configurations.configurations import CFG
    config = Config.from_json(CFG)
    glove_dir = config.external_data_sources.word_embeddings
    file_name = config.external_data_sources.embeddings_file_name

    def __init__(self):
        self.config = Config.from_json(CFG)
        self.texts = []
        self.labels = []
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
                    self.texts.append(f.read())  # The text of each .txt appends to a universal structure
                    f.close()
                    if label_type == 'neg':  # The labels of each .txt appends to a universal structure
                        self.labels.append(0)
                    else:
                        self.labels.append(1)

    # Counts the words with an incremental index and informs the dictionary of tokenizer
    # <word , index_number>
    def indexing_informs_tokenizer(self):
        self.tokenizer.fit_on_texts(self.texts)  # update vocabulary
        word_index = self.tokenizer.word_index  # create integer indices pre token.
        print('Found %s unique tokens.' % len(word_index))
        return word_index

    def shape_tensors_and_store_data(self):
        from keras.preprocessing.sequence import pad_sequences  # to convert the lists into 2D of same sizes.
        sequences = self.tokenizer.texts_to_sequences(self.texts)  # convert text to numbers on most frequent words
        # each list of words represented by a sequential numbers
        # every list must have the same size, zero paddings to fill
        data = pad_sequences(sequences, maxlen=self.max_len)
        labels = np.asarray(self.labels)
        print('Shape of data tensor:', data.shape)
        print('Shape of label tensor:', labels.shape)
        # incremental list of size of data with the indices for shuffling the words.
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        # the words shuffled randomly and we store them
        data = data[indices]
        labels = labels[indices]
        return data, labels

    def split_data(self):
        x, y = self.shape_tensors_and_store_data()
        return x[:self.training_samples], \
               y[:self.training_samples], \
               x[self.training_samples: self.training_samples + self.validation_samples], \
               y[self.training_samples: self.training_samples + self.validation_samples]

    # HDF5 have the pros of much faster I/O and compressed size than
    # SQL storage and the dis of memory vs the solid storage.
    def store_h5py(self, x_train, y_train, x_val, y_val):
        try:
            from h5py import File
            hdf = File(self.config.data.HDFS_INTERNAL_DATA_FILENAME, "w")
        except IOError:
            print("The internal file failed to open for write.")
        else:
            group_data = hdf.create_group("dataset")

            group_x_train = group_data.create_group("train/x_train")
            group_y_train = group_data.create_group("train/y_train")
            group_x_val = group_data.create_group("val/x_val")
            group_y_val = group_data.create_group("val/y_val")

            group_x_train.create_dataset("x_trainset",data=x_train,compression="gzip")
            group_y_train.create_dataset("y_trainset",data=y_train,compression="gzip")

            group_x_val.create_dataset("x_valset",data=x_val,compression="gzip")
            group_y_val.create_dataset("y_valset",data=y_val,compression="gzip")
            hdf.close()

        return
