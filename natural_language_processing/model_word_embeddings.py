"""Not enough data available to learn truly
powerful features on your own, but you expect the features that you need to be fairly
generic—that is. In this case, it makes
sense to reuse features learned on a different problem, so use pretrained Word Embedding
from GloVe or Word2Vec. If the data are a few if they are not the problem is not so generic and
pretrained embedding have worse result because of their generalization.
"""

from natural_language_processing.configurations.configuration_infrastructure import Config
from natural_language_processing.configurations.configurations import CFG
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

# class Model():

config = Config.from_json(CFG)
embeddings_dimension = config.external_data_sources.embeddings_dimension
max_words = config.data.max_words
max_len = config.data.max_words
hidden_layers_nodes = config.model.hidden_layers_nodes
last_layer_nodes = config.model.last_layer_nodes
hidden_layers_act_func = config.model.hidden_layers_act_func
last_layer_act_func = config.model.last_layer_activation_function

model = Sequential()
model.add(Embedding(max_words, embeddings_dimension, input_length=max_len))
model.add(Flatten())
model.add(Dense(units=hidden_layers_nodes, activation=hidden_layers_act_func))  # implements 32 outputs = activation(dot(input, kernel) + bias)
model.add(Dense(units=last_layer_nodes, activation=last_layer_act_func))
model.summary()


# load the word embeddings in the model
from natural_language_processing.parse_word_embeddings import ParseWordEmbeddings
from natural_language_processing.text_processing import TextProcessing


data_proc = TextProcessing()
data_proc.process_train_data()
data_proc.shape_tensors_and_store_data()
x_train, y_train, x_val, y_val = data_proc.split_data()
import numpy as np
print ("shape: ", np.shape(x_train))
print ("shape: ", np.shape(y_train))
print ("shape: ", np.shape(x_val))
print ("shape: ", np.shape(y_val))
ParseWordEmbeddings.embeddings_vectors()

embeddings_matrix = ParseWordEmbeddings.create_embeddings_matrix(data_proc.indexing_informs_tokenizer())  # weights of the neural network
model.layers[0].set_weights([embeddings_matrix])
# Freeze the embeddings layer, pretrained parts should not be updated
# to forget what they earned
model.layers[0].trainable = False

import tensorflow as tf
#Configures the model for training.
model.compile(optimizer='rmsprop'
# tf.keras.optimizers.RMSprop(
#                 learning_rate=0.001
#                 , rho=0.9
#                 , momentum=0.0
#                 , epsilon=1e-7
#                 , centered=False
#                 , name='RMSprop')  # for more that 10k records use mini-batches, rmsprop tf.keras.optimizers.RMSprop
              ,loss=config.model.loss_function,
              metrics=config.model.metrics,
              loss_weights=None,
              weighted_metrics=None,
              run_eagerly=None)

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]

history = model.fit(x_train,
                    y_train,
                    epochs=config.train.epochs,
                    batch_size=config.train.batch_size,
                    validation_data=(x_val, y_val), # <----- here throws the Value error
                    callbacks=my_callbacks)

model.save_weights('pretrained_model.h5')