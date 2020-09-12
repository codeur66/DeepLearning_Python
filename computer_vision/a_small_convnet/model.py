# Conv2D and MaxPooling2D layers
# Listing 5.1
from keras import layers
from keras import models

# Sequential groups a linear stack of layers into a `tf.keras.Model`.
# Sequential provides training and inference features on this model.
model = models.Sequential()

# Images are in RGB => three features => three layers.

model.add(layers.Conv2D(32, (3, 3)), activate='relu', input_shape=(28, 28, 1))
# MaxPool2D: It is basically used to reduce the size of the image because the larger number
# of pixels contribute to more parameters which can involve large chunks of data.
# Thus we need less parameters such that a CNN can still identify the image.
# or choose the Average value
model.add(layers.MaxPool2D(2, 2))
print("for debug")
model.add(layers.Conv2D())
model.add(layers.MaxPool2D())
model.add(layers.Conv2D())
model.add(layers.MaxPool2D())
print("for debug")
