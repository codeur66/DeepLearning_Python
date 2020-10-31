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

from .BaseModel import BaseModel

class Model(BaseModel):
    config = Config.from_json(CFG)
    embeddings_dimension = config.external_data_sources.embeddings_dimension
    max_words = config.data.max_words
    max_len = config.data.maxlen
    hidden_layers_nodes = config.model.hidden_layers_nodes
    last_layer_nodes = config.model.last_layer_nodes
    hidden_layers_act_func = config.model.hidden_layers_act_func
    last_layer_act_func = config.model.last_layer_activation_function

    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = None

    def load_data(self, **kwargs):
        # get the data and call the method of processing
        data_proc = TextProcessing()
        data_proc.process_train_data()
        data_proc.shape_tensors_and_store_data()
        x_train, y_train, x_val, y_val = data_proc.split_data()
        # get the pretrained data
        ParseWordEmbeddings.embeddings_vectors()
        word_index = data_proc.indexing_informs_tokenizer()
        # get the pretrainned weights to insert into the neural network
        embeddings_matrix = ParseWordEmbeddings.create_embeddings_matrix(word_index)
        return x_train, y_train, x_val, y_val, embeddings_matrix

    def build_architecture(self):
        self.model = Sequential()
        self.model.add(Embedding(Model.max_words, Model.embeddings_dimension, input_length=Model.max_len))
        self.model.add(Flatten())
        self.model.add(Dense(32, activation='relu'))  # implements 32 outputs = activation(dot(input, kernel) + bias)
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.summary()
        plot_model(self.model, show_shapes=True)
        return self

    def build(self, **kwargs):
        self.model.layers[0].set_weights([embeddings_matrix])
        # Freeze the embeddings layer, pretrained parts should not be updated to forget what they earned
        self.model.layers[0].trainable = False

        # Configures the model for training.
        self.model.compile(optimizer=Model.config.model.optimizer[0],
                           loss=Model.config.model.loss_function,
                           metrics=Model.config.model.metrics,
                           loss_weights=None,
                           weighted_metrics=None)

    def train(self, **kwargs):
        # for debugging the model
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
        from os import getcwd
        cwd = getcwd()
        self.model.save_weights(
            "/home/nikoscf/PycharmProjects/DeepLearningWithPython/natural_language_processing/log_model/trained_model.h5")
        return history

    def evaluate(self, **kwargs):
            import matplotlib.pyplot as plt
            acc = hist.history["acc"]
            val_acc = hist.history['val_acc']
            loss = hist.history['loss']
            val_loss = hist.history['val_loss']
            epochs = range(1, len(acc) + 1)
            plt.plot(epochs, acc, 'bo', label='Training acc')
            plt.plot(epochs, val_acc, 'b', label='Validation acc')
            plt.title('Training and validation accuracy')
            plt.legend()
            plt.figure()
            plt.plot(epochs, loss, 'bo', label='Training loss')
            plt.plot(epochs, val_loss, 'b', label='Validation loss')
            plt.title('Training and validation loss')
            plt.legend()
            plt.show()

md = Model()
x_train, y_train, x_val, y_val, embeddings_matrix = md.load_data()
md.build_architecture().build()
hist = md.train(x_train = x_train, y_train = y_train, x_val = x_val, y_val = y_val, embeddings_matrix = embeddings_matrix)
eval_plot = md.evaluate(history = hist)


