
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import to_categorical
from computer_vision.a_small_convnet.model.model import model_path

model = load_model(model_path, compile=False)

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32')/255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'],
              loss_weights=None,
              weighted_metrics=None,
              run_eagerly=None)

model.fit(x=train_images,
          y=train_labels,
          batch_size=64,
          epochs=5
          )

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("test_loss : {}, test accuracy : {}", format(test_loss, test_acc))
