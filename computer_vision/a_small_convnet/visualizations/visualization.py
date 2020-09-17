from computer_vision.a_small_convnet.train.train import Train
from keras.models import load_model
from keras.preprocessing import image
import numpy as np


model = load_model(Train.model_path('simple_convnet.h5'), compile=False) # False, tricks keras to avoid compilation
