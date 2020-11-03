"""If the data are a few and the problem is quite specific
and pretrained embeddings have worse result because of their generalizations.
There are not enough data available to learn truly
powerful features, but you expect the features that you need to be fairly
generic reuse features learned on a different problem, so use pretrained Word Embeddings
from GloVe or Word2Vec.
"""

from natural_language_processing.configurations.configuration_infrastructure import Config
from natural_language_processing.configurations.configurations import CFG
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.utils import plot_model
# load the word embeddings in the model
from natural_language_processing.data.parse_word_embeddings import ParseWordEmbeddings
from natural_language_processing.data.text_processing import TextProcessing
from natural_language_processing.model.BaseModel import BaseModel


class Model(BaseModel):
    config = Config.from_json(CFG)
    embeddings_dimension = config.external_data_sources.embeddings_dimension
    max_words = config.data.max_words
    max_len = config.data.maxlen
    hidden_layers_nodes = config.model.hidden_layers_nodes
    last_layer_nodes = config.model.last_layer_nodes
    hidden_layers_act_func = config.model.hidden_layers_act_func
    last_layer_act_func = config.model.last_layer_activation_function
    path_model = config.model.path_model

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
        # ParseWordEmbeddings.embeddings_vectors()
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
        plot_model(self.model, to_file=Model.path_model+"model.png", show_shapes=True)
        return self

    def build(self, **kwargs):
        self.model.layers[0].set_weights([embeddings_matrix])
        # Freeze the embeddings layer, pretrained parts should not be updated to forget what they learned
        self.model.layers[0].trainable = False
        # Configures the model for training.
        self.model.compile(optimizer="rmsprop",
                           loss=Model.config.model.loss_function,
                           metrics=Model.config.model.metrics,
                           loss_weights=None,
                           weighted_metrics=None)

    def train(self, **kwargs):
        history = self.model.fit(x_train,
                                 y_train,
                                 epochs=Model.config.train.epochs,
                                 batch_size=Model.config.train.batch_size,
                                 validation_data=(x_val, y_val))
        self.model.save_weights(Model.path_model+"trained_model.h5")
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


md = Model(CFG)
x_train, y_train, x_val, y_val, embeddings_matrix = md.load_data()
md.build_architecture().build()
hist = md.train(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, embeddings_matrix=embeddings_matrix)
eval_plot = md.evaluate(history=hist)
