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
import tensorflow as tf
from keras.utils import plot_model
# load the word embeddings in the model
from natural_language_processing.parse_word_embeddings import ParseWordEmbeddings
from natural_language_processing.text_processing import TextProcessing


class Model:
    config = Config.from_json(CFG)
    embeddings_dimension = config.external_data_sources.embeddings_dimension
    max_words = config.data.max_words
    max_len = config.data.maxlen
    hidden_layers_nodes = config.model.hidden_layers_nodes
    last_layer_nodes = config.model.last_layer_nodes
    hidden_layers_act_func = config.model.hidden_layers_act_func
    last_layer_act_func = config.model.last_layer_activation_function

    def __init__(self):
        self.model = None

    def build_architecture(self):
        self.model = Sequential()
        self.model.add(Embedding(Model.max_words, Model.embeddings_dimension, input_length=Model.max_len))
        self.model.add(Flatten())
        self.model.add(Dense(32, activation='relu'))  # implements 32 outputs = activation(dot(input, kernel) + bias)
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.summary()
        plot_model(self.model, show_shapes=True)
        return self

    def train(self):
        # get the data and call the method of processing
        data_proc = TextProcessing()
        data_proc.process_train_data()
        data_proc.shape_tensors_and_store_data()
        x_train, y_train, x_val, y_val = data_proc.split_data()
        # get the pretrained data
        ParseWordEmbeddings.embeddings_vectors()
        word_index = data_proc.indexing_informs_tokenizer()
        # get the pretrainned weights to insert into the neural network
        embeddings_matrix = ParseWordEmbeddings.create_embeddings_matrix(
            word_index)

        self.model.layers[0].set_weights([embeddings_matrix])
        # Freeze the embeddings layer, pretrained parts should not be updated to forget what they earned
        self.model.layers[0].trainable = False

        # Configures the model for training.
        self.model.compile(optimizer='rmsprop',
                           loss=Model.config.model.loss_function,
                           metrics=Model.config.model.metrics,
                           loss_weights=None,
                           weighted_metrics=None)
        # for debugging log the model
        model_callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=2),
            tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
            tf.keras.callbacks.TensorBoard(log_dir='./logs')
        ]

        history = self.model.fit(x_train,
                                 y_train,
                                 epochs=Model.config.train.epochs,
                                 batch_size=Model.config.train.batch_size,
                                 validation_data=(x_val, y_val),
                                 callbacks=model_callbacks)

        self.model.save_weights('pretrained_model.h5')


md = Model()
md.build_architecture().train()