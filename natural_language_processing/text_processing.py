import numpy as np


# Word-level one-hot encoding
def word_level():
    samples = ['The cat sat on the mat.', 'The dog ate my homework.']
    token_index = {}
    for sample in samples:
        for word in sample.split():
            if word not in token_index:
                token_index[word] = len(token_index) + 1
    max_length = 10
    results = np.zeros(shape=(len(samples), max_length, max(token_index.values()) + 1))

    for i, sample in enumerate(samples):
        for j, word in list(enumerate(sample.split()))[:max_length]:
            index = token_index.get(word)
            results[i, j, index] = 1.


# Character-level one-hot encoding
def character_level():
    import string
    samples = ['The cat sat on the mat.', 'The dog ate my homework.']
    characters = string.printable
    print("characters ; ", characters)
    token_index = dict(zip(range(1, len(characters) + 1), characters))
    print("token_index ; ", token_index)
    max_length = 50
    results = np.zeros((len(samples), max_length, max(token_index.keys()) + 1))
    for i, sample in enumerate(samples):
        for j, character in enumerate(sample):
            print("character ; ", character)
            index = token_index.get(character)
            results[i, j, index] = 1.
    print("character_level : ", results)


# keras build in utilities for one hot encoding
def keras_tokenizer():
    from keras.preprocessing.text import Tokenizer
    samples = ['The cat sat on the mat.', 'The dog ate my homework.']
    tokenizer = Tokenizer(num_words=1000) # only take into account the 1.000 most common words
    tokenizer.fit_on_texts(samples) # builds the word index
    sequences = tokenizer.texts_to_sequences(samples)   # Turns strings into lists of integer indices
    print("sequences : ", sequences)
    one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
    print("one_hot_results : ", one_hot_results)
    word_index = tokenizer.word_index # recover the word index that was computed
    print('Found %s unique tokens.' % len(word_index))
"""
word representations
obtained from one-hot encoding or hashing are
sparse, high-dimensional, and hardcoded, word
embeddings are dense, relatively low-
dimensional, and learned from data.
"""
def word_embeddings():
    from keras.layers import Embedding
    # when we instantiate an Embedding layer, its weights are random,
    # During training, these word vectors are gradually adjusted via
    # backpropagation, structuring the space.
    embedding_layer = Embedding(1000, 64)

word_level()
character_level()
keras_tokenizer()
word_embeddings()