from keras.datasets import mnist
from keras.utils import to_categorical
from computer_vision.a_small_convnet.model.model import Model
from os import getcwd


class Train:

    def __init__(self, model_architecture):
        self.model_architecture = model_architecture
        self.optimizer = 'rmsprop'
        self.loss = 'categorical_crossentropy'
        self.metrics = ['accuracy']
        self.loss_weights = None
        self.weighted_metrics = None
        self.run_eagerly = None
        self.batch_size = 64
        self.epochs = 1
        self.train_images, self.train_labels, self.test_images, self.test_labels = \
            self.reshape_imgs()

    @classmethod
    def train_test_data(cls):
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        return (train_images, train_labels), (test_images, test_labels)

    @staticmethod
    def reshape_imgs():
        (train_images, train_labels), (test_images, test_labels) = Train.train_test_data()

        train_images = train_images.reshape((60000, 28, 28, 1))
        train_images = train_images.astype('float32') / 255
        test_images = test_images.reshape((10000, 28, 28, 1))
        test_images = test_images.astype('float32') / 255
        train_labels = to_categorical(train_labels)
        test_labels = to_categorical(test_labels)

        return train_images, train_labels, test_images, test_labels

    def _set_training_parameters(self):
        self.model_architecture.model.compile(optimizer=self.optimizer,
                                              loss=self.loss,
                                              metrics=self.metrics,
                                              loss_weights=self.loss_weights,
                                              weighted_metrics=self.weighted_metrics,
                                              run_eagerly=self.run_eagerly)

    def training(self):
        self._set_training_parameters()
        self.model_architecture.model.fit(x=self.train_images,
                                          y=self.train_labels,
                                          batch_size=self.batch_size,
                                          epochs=self.epochs
                                          )

    def evaluate(self):
        test_loss, test_acc = self.model_architecture.model.evaluate(self.test_images,
                                                                     self.test_labels)
        print("test_loss : {}, test accuracy : {}".format(test_loss, test_acc))

    # input to visualization, does not work if invoke the saved model
    # from a foreign-next module backwards to this.
    def save_model(self, model_name):
        self.model_architecture.model.save(model_name, overwrite = True)


    @classmethod
    def model_path(cls, model_name):

        return getcwd() + '/' + model_name


def exec_training(model_name):
    train.training()
    train.evaluate()
    train.save_model(model_name)


def load_model(model_name):
    from keras.models import load_model
    return load_model(model_name)


model_design = Model()
train = Train(model_design)
exec_training('simple_convnet.h5')

