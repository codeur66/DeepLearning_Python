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
from natural_language_processing.model.base_model import BaseModel


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

    def __init__(self, cfg, hdf_in_flag, hdf_ext_flag):
        super().__init__(cfg)
        self.model = None
        self.hdf_in_flag = hdf_in_flag
        self.hdf_ext_flag = hdf_ext_flag
        self.hdf_in = None
        self.hdf_ext = None

    def load_data(self, **kwargs):
        import natural_language_processing.model.read_hdf5 as rd
        if (self.hdf_in_flag, self.hdf_ext_flag) == (True, False):
            x_trn, y_trn, x_valid, y_valid, x_tst, y_tst, self.hdf_in = rd.get_internal_hdf()
            return x_trn[:], y_trn[:], x_valid[:], y_valid[:], x_tst[:], y_tst[:], None
        elif (self.hdf_in_flag, self.hdf_ext_flag) == (True, True):
            x_trn, y_trn, x_valid, y_valid, x_tst, y_tst, self.hdf_in = rd.get_internal_hdf()
            embeddings, self.hdf_ext = rd.get_external_hdf()
            return x_trn[:], y_trn[:], x_valid[:], y_valid[:], x_tst[:], y_tst[:], embeddings[:]
        else:
            print("Did not provided appropriate datasets' choices for the model.")

    def build_architecture(self):
        self.model = Sequential()
        self.model.add(Embedding(Model.max_words, Model.embeddings_dimension, input_length=Model.max_len))
        self.model.add(Flatten())
        self.model.add(Dense(32, activation='relu'))  # implements 32 outputs = activation(dot(input, kernel) + bias)
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.summary()
        plot_model(self.model, to_file=Model.path_model + "model.png", show_shapes=True)
        return self

    def build(self, **kwargs):
        # Freeze the embeddings if we choose world embeddings
        if self.hdf_ext_flag is True:
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
        try:
            history = self.model.fit(x_train,
                                     y_train,
                                     epochs=Model.config.train.epochs,
                                     batch_size=Model.config.train.batch_size,
                                     validation_data=(x_val, y_val))
            self.model.save_weights(Model.path_model + "trained_model.h5")

            return history
        except RuntimeError:  # if memory crashes
            raise RuntimeError

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
        # evaluate on test data
        self.model.load_weights(Model.path_model+'trained_model.h5')
        self.model.evaluate(x_test, y_test)

    def close_files(self):
        try:
            self.hdf_in.close()
            self.hdf_ext.close()
        except IOError:
            raise Exception("Failed to close the open files of datasets ")


if __name__ == '__main__':
    use_internal_data = True
    use_embeddings_data = False
    md = Model(CFG, use_internal_data, use_embeddings_data)
    x_train, y_train, x_val, y_val, x_test, y_test, embeddings_matrix = md.load_data()
    md.build_architecture().build(embeddings_matrix=embeddings_matrix)
    hist = md.train(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
    md.evaluate(history=hist, x_test=x_test, y_test=y_test)
