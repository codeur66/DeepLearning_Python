
import tensorflow.keras.optimizers
CFG = {
    "external_data_sources": {
        "word_embeddings": "/home/nikoscf/PycharmProjects/DeepLearningWithPython/natural_language_processing/data/glove.6B"
        , "embeddings_file_name": "glove.6B.100d.txt"
        , "embeddings_dimension": 100  # embedding vectors must = unique tokens (weights)
    },
    "data": {
        "path_train_data": '/home/nikoscf/PycharmProjects/DeepLearningWithPython/natural_language_processing/data/aclImdb/aclImdb'
        , "maxlen": 100
        , "training_samples": 100 # ! see (A) at the end
        , "validation_samples": 2000
        , "max_words": 10000

    },
    "model": {
        "hidden_layers_nodes": 32
        , "last_layer_nodes": 1
        , "hidden_layers_act_func": 'relu'
        , "last_layer_activation_function": 'sigmoid' # for binary classification we choose sigmoid activation function
        , "loss_function": "binary_crossentropy"
        , "metrics": ["acc"]
        , "path_model": "/home/nikoscf/PycharmProjects/DeepLearningWithPython/natural_language_processing/model/log_model/"

    },
    "train": {
          "epochs": 20
        , "batch_size": 32
    }
}


""" (A) 
Choice of training_samples":100, 200, 2000 observer the overfitting from Embeddings
 to the specific task .
Pretrained word embeddings are meant to
be particularly useful on problems where little training data is available (otherwise,
task-specific embeddings are likely to outperform them).We restrict the learning
to only 200 RANDOM unique records to use the word embeddings.
If this works poorly, try choosing a different random set of 200 samples for the exercise.
 """

# momentum=0.1 # default =0, for mini-batches experiment with the direction
# learning_rate = 0.001  # Don’t turn down the learning rate too soon, much later increases the error
# tf.keras.optimizers.RMSprop(  # for more that 10k records use mini-batches: rmsprop(use of moving averaging)
