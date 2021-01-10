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
import os.path
from natural_language_processing.logging.LoggerCls import LoggerCls


class Model(BaseModel):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    formatter = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    logModel = LoggerCls("log_to_file", "Model", dir_path + "/Model.log", "w", formatter, "INFO")
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
        try:
            if (self.hdf_in_flag, self.hdf_ext_flag) == (True, False):
                x_trn, y_trn, x_valid, y_valid, x_tst, y_tst, self.hdf_in = rd.get_internal_hdf()
                return x_trn[:], y_trn[:], x_valid[:], y_valid[:], x_tst[:], y_tst[:], None
            elif (self.hdf_in_flag, self.hdf_ext_flag) == (True, True):
                x_trn, y_trn, x_valid, y_valid, x_tst, y_tst, self.hdf_in = rd.get_internal_hdf()
                embeddings, self.hdf_ext = rd.get_external_hdf()
                return x_trn[:], y_trn[:], x_valid[:], y_valid[:], x_tst[:], y_tst[:], embeddings[:]
            else:
                Model.logModel.error(
                    "Did not provided an appropriate choice for the model to choose the training datasets.")
                # print("Did not provided an appropriate choice for the model to choose the training datasets.")
        except (TypeError, AttributeError, RuntimeError) as e:
            Model.logModel.error(e)
            # raise Model.logModel.error(
            #     "Did not provided an appropriate choice for the model to choose the training datasets.")

    def build_architecture(self):
        try:
            Model.logModel.info("The model sets the network architecture")
            self.model = Sequential()
            self.model.add(Embedding(Model.max_words, Model.embeddings_dimension, input_length=Model.max_len))
            self.model.add(Flatten())
            # 32 outputs with activation(dot(input, kernel) + bias)
            self.model.add(Dense(32, activation='relu'))
            self.model.add(Dense(1, activation='sigmoid'))
            self.model.summary()
            plot_model(self.model, to_file=Model.path_model + "model.png", show_shapes=True)
            return self
        except (TypeError, AttributeError, RuntimeError) as e:
            Model.logModel.error("Error encountered while the model try to build the architecture")
            Model.logModel.error(e)

    def build(self, **kwargs):
        try:
            Model.logModel.info("Configuration of the hyperparameters")
            # Freeze the embeddings if we choose world embeddings
            if self.hdf_ext_flag is True:
                self.model.layers[0].set_weights([kwargs['embeddings_matrix']])
                # Freeze the embeddings layer, pretrained parts should not be updated to forget what they learned
                self.model.layers[0].trainable = False

            # Configures the model for training.
            self.model.compile(optimizer="rmsprop",
                               loss=Model.config.model.loss_function,
                               metrics=Model.config.model.metrics,
                               loss_weights=None,
                               weighted_metrics=None)
        except (TypeError, AttributeError, RuntimeError) as e:
            Model.logModel.error("Error encountered while the model configures the architecture")
            Model.logModel.error(e)

    def train(self, **kwargs):
        try:
            Model.logModel.info("Starting the model training.")
            History = self.model.fit(kwargs['x_train'],
                                     kwargs['y_train'],
                                     epochs=Model.config.train.epochs,
                                     batch_size=Model.config.train.batch_size,
                                     validation_data=(kwargs['x_val'], kwargs['y_val']))
            self.model.save_weights(Model.path_model + "trained_model.h5")
            return History
        except (TypeError, AttributeError, RuntimeError) as e:
            Model.logModel.error("Error on training")
            Model.logModel.error(e)

    def evaluate(self, **kwargs):
        try:
            Model.logModel.info("Starts to create the evaluation plots of the model.")
            import matplotlib.pyplot as plt
            acc = kwargs['History'].history["acc"]
            val_acc = kwargs['History'].history['val_acc']
            loss = kwargs['History'].history['loss']
            val_loss = kwargs['History'].history['val_loss']
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
            self.model.load_weights(Model.path_model + 'trained_model.h5')
            eval_metrics = self.model.evaluate(kwargs['x_test'], kwargs['y_test'])
            Model.logModel.info('loss, acc:'  + str(eval_metrics))
        except (TypeError, AttributeError, RuntimeError, ValueError) as e:
            Model.logModel.error("Error encountered on the model evaluation.")
            Model.logModel.error(e)

    def close_files(self):
        try:
            self.hdf_in.close()
            self.hdf_ext.close()
        except IOError as e:
            Model.logModel.error("Failed to close the open files of datasets ")
            Model.logModel.error(e)


if __name__ == '__main__':
    def exec_model_pipeline(use_internal_data, use_embeddings_data):

        md = Model(CFG, use_internal_data, use_embeddings_data)
        x_train, y_train, x_val, y_val, x_test, y_test, embeddings_matrix = md.load_data()

        md.build_architecture().build(embeddings_matrix=embeddings_matrix)
        hist = md.train(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)

        md.evaluate(History=hist, x_test=x_test, y_test=y_test)


    exec_model_pipeline(use_internal_data=True, use_embeddings_data=False)
