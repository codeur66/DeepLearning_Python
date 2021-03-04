from natural_language_processing.configurations.configuration_infrastructure import Config
from natural_language_processing.configurations.configurations import CFG
import os.path
from natural_language_processing.logging.LoggerCls import LoggerCls


class ModelConfigurations:

    def __init__(self):
        super().__init__()

        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.formatter = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        self.logModel = LoggerCls("log_to_file", "Model", self.dir_path + "/Model.log", "w", self.formatter, "INFO")

        self.config = Config.from_json(CFG)
        self.embeddings_dimension = self.config.external_data_sources.embeddings_dimension
        self.max_words = self.config.data.max_words
        self.max_len = self.config.data.maxlen
        self.hidden_layers_nodes = self.config.model.hidden_layers_nodes
        self.last_layer_nodes = self.config.model.last_layer_nodes
        self.hidden_layers_act_func = self.config.model.hidden_layers_act_func
        self.last_layer_act_func = self.config.model.last_layer_activation_function
        self.path_model = self.config.model.path_model
