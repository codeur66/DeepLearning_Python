
class SentimentDataProcessing():

    def __init__(self):
        from natural_language_processing.configurations.configuration_infrastructure import Config
        from natural_language_processing.configurations.configurations import CFG
        self.config = Config.from_json(CFG)
        self.texts, self.labels = self.process_train_data()

    # Reads the folders stores in data structures the corresponding data (data and labels) for the train.
    def process_train_data(self):
        import os
        train_dir = os.path.join(self.config.train.path_train_data, 'train')
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

    """Pretrained word embeddings are meant to
    be particularly useful on problems where little training data is available (otherwise,
    task-specific embeddings are likely to outperform them), so we restrict the learning 
    to only 200 unique records."""

    def tokenize_train_data(self):
        from keras.preprocessing.text import Tokenizer
        from keras.preprocessing.sequence import pad_sequences  # to make the lists insto 2D of same sizes.
        import numpy as np
        maxlen = 100  # cut of the reviews after 100 words
        training_samples = 200
        validation_samples = 10000
        max_words = 10000  # keep only the first 10.000 words of dataset
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(self.texts)
        sequences = tokenizer.texts_to_sequences(self.texts)
        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        data = pad_sequences(sequences, maxlen=maxlen)
        labels = np.asarray(self.labels)
        print('Shape of data tensor:', data.shape)
        print('Shape of label tensor:', labels.shape)
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]


data =  SentimentDataProcessing()
train_data, labels = data.process_train_data()
tokens = data.tokenize_train_data()
