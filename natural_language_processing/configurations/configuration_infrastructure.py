import json

class Config:
    """Config class which contains data, train and model hyperparameters"""

    def __init__(self, external_data_sources, data, train, model):
        self.external_data_sources = external_data_sources
        self.data = data
        self.train = train
        self.model = model

    def default(self, object):
        return  json.JSONEncoder.default()

    @classmethod
    def from_json(cls, cfg):
        """Creates config from json"""
        import tensorflow.keras.optimizers  as tf
        # params = json.loads(json.dumps(cfg, default = lambda o: o.__dict__), object_hook=HelperObject)
        params = json.loads(json.dumps(cfg), object_hook=HelperObject)
        return cls(params.external_data_sources, params.data, params.train, params.model)


class HelperObject(object):
    """Helper class to convert json into Python object"""
    def __init__(self, dict_):
        self.__dict__.update(dict_)



