import numpy as np
from natural_language_processing.configurations.configuration_infrastructure import Config
from natural_language_processing.configurations.configurations import CFG
from keras.preprocessing.text import Tokenizer

"""TextProcessing, different data sources
may be added in the pipeline for other models or the current may need extension.
In contrast to ParseWordEmbeddings class which holds external data source - pretrained.
"""
from natural_language_processing.logging.LoggerCls import LoggerCls
import os.path


class TextProcessing:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    formatter = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    logToFile = LoggerCls("log_to_file", " data pipeline processor:", dir_path + "/data_pipeline.log", "w", formatter,
                          "INFO")
    logToStream = LoggerCls("log_to_stdout", "data pipeline processor: ", None, "w", formatter, "INFO")

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
        self.tokenizer = Tokenizer(num_words=self.max_words) # the max numer of words to keep, the same for train and tests

    # Reads the folders stores in data structures the corresponding data (data and labels) for the train.

    def process_train_tst_data(self, path_data, filename):
        import os
        data_dir = os.path.join(path_data, filename)
        for label_type in ['neg', 'pos']:
            dir_name = os.path.join(data_dir, label_type)
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
        TextProcessing.logToStream.info('Found %s unique tokens.' % len(word_index))
        return word_index

    def shape_tensors_and_store_data(self):
        from keras.preprocessing.sequence import pad_sequences  # to convert the lists into 2D of same sizes.
        sequences = self.tokenizer.texts_to_sequences(self.texts)  # convert text to numbers on most frequent words
        # each list of words represented by a sequential numbers
        # every list must have the same size, zero paddings to fill
        data = pad_sequences(sequences, maxlen=self.max_len)
        labels = np.asarray(self.labels)
        # TextProcessing.logToStream.info('Shape of data  tensor : %s' % data.shape)
        # TextProcessing.logToStream.info('Shape of label tensor : %s' % labels.shape)

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
    def store_h5py(self, x_train, y_train, x_val, y_val, x_test, y_test):
        try:
            from h5py import File
            hdf = File(self.config.data.HDFS_INTERNAL_DATA_FILENAME, "w")
        except IOError as e :
            TextProcessing.logToFile.error("The internal file failed to open for write in <TextProcessing/store_h5py")
            TextProcessing.logToFile.error(e)
        else:
            try:
                group_data = hdf.create_group("dataset")

                group_train = group_data.create_group("train")
                group_val = group_data.create_group("val")
                group_test = group_data.create_group("test")

                group_train.create_dataset("x_trainset", data=x_train, compression="gzip")
                group_train.create_dataset("y_trainset", data=y_train, compression="gzip")

                group_val.create_dataset("x_valset", data=x_val, compression="gzip")
                group_val.create_dataset("y_valset", data=y_val, compression="gzip")

                group_test.create_dataset("x_testset", data=x_test, compression="gzip")
                group_test.create_dataset("y_testset", data=y_test, compression="gzip")
            except IOError as e:
                TextProcessing.logToFile.error("Failed to store in hdfs in <TextProcessing/store_h5py")
                TextProcessing.logToFile.error(e)
            else:
                hdf.close()
                TextProcessing.logToFile.logger.info("Successful creation of data file with in-house text processing.")
                TextProcessing.logToStream.logger.info("Successful creation of data file with in-house text processing.")

