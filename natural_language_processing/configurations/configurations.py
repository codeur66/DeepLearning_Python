import tensorflow as tf

CFG = {
    "external_data_sources": {
        "word_embeddings": "/home/nikoscf/PycharmProjects/DeepLearningWithPython/natural_language_processing/data/glove.6B"
        , "embeddings_file_name": "glove.6B.100d.txt"
        , "embeddings_dimension": 100  # embedding vectors
    },
    "data": {
        "path_train_data": '/home/nikoscf/PycharmProjects/DeepLearningWithPython/natural_language_processing/data/aclImdb/aclImdb'
        , "maxlen": 100
        , "training_samples": 200 # see (A) at the end
        , "validation_samples": 10000
        , "max_words": 10000

    },
    "model": {
        "hidden_layers_nodes": 32
        , "last_layer_nodes": 1
        , "hidden_layers_act_func": 'relu'
        , "last_layer_activation_function": 'sigmoid' # for binary classification we choose sigmoid activation function
        , "optimizer": [
           """ tf.keras.optimizers.RMSprop(
                learning_rate=0.001  
                , rho=0.9
                , momentum=0.1 
                , epsilon=1e-7
                , centered=False 
                , name='RMSprop')"""
            , """tf.keras.optimizers.SGD(
                learning_rate=0.01, momentum=0.0, nesterov=False, name="SGD")"""
        ]
        , "loss_function": "binary_crossentropy"
        , "metrics": ["acc"]

    },
    "train": {
          "epochs": 10
        , "batch_size": 32
    }
}


""" (A) 
Choice of training_samples":200 
Pretrained word embeddings are meant to
be particularly useful on problems where little training data is available (otherwise,
task-specific embeddings are likely to outperform them).We restrict the learning
to only 200 unique records to use the word embeddings. """

# momentum=0.1 # default =0, for mini-batches experiment with the direction
# learning_rate = 0.001  # Don’t turn down the learning rate too soon, much later increases the error
# tf.keras.optimizers.RMSprop(  # for more that 10k records use mini-batches: rmsprop(use of moving averaging)

