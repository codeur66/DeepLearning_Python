from computer_vision.a_small_convnet.train.train import model_train_path
from keras.models import load_model
from keras.preprocessing import image
import numpy as np


_model = load_model(model_train_path, compile=False) # False, tricks keras to avoid compilation
