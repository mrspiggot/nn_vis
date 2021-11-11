import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Helper libraries
import numpy as np
from datetime import datetime
import pickle
from sklearn.metrics import confusion_matrix

class MNISTBase():
    def __init__(self, n="numbers"):
        self.name = n
        if n == 'fashion':
            self.class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
            (self.train_images, self.train_labels), (self.test_images, self.test_labels) = self.load_fashion()

        elif n == 'cifar10':
            self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
            (self.train_images, self.train_labels), (self.test_images, self.test_labels) = self.load_cifar10()

        else:
            self.class_names = ['Zero', 'One', 'Two', 'Three', 'Four',
               'Five', 'Six', 'Seven', 'Eight', 'Nine']
            (self.train_images, self.train_labels), (self.test_images, self.test_labels) = self.load_numbers()

    def load_fashion(self):
        fashion_mnist = tf.keras.datasets.fashion_mnist
        return fashion_mnist.load_data()

    def load_numbers(self):
        numbers_mnist = tf.keras.datasets.mnist
        return numbers_mnist.load_data()

    def load_cifar10(self):
        cifar10_mnist = tf.keras.datasets.cifar10
        return cifar10_mnist.load_data()


class MLP():
    def __init__(self, mnist, hidden_layers, activation):
        self.mnist = mnist
        self.num_layers = len(hidden_layers)
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.model = self.build_model()


    def build_model(self):
        t_s = self.mnist.train_images.shape

        if len(t_s) > 3:
            self.mnist.train_images = tf.image.rgb_to_grayscale(self.mnist.train_images)
            self.mnist.test_images = tf.image.rgb_to_grayscale(self.mnist.test_images)

        layer_list = [tf.keras.layers.Flatten(input_shape=t_s[1:3])]
        for number_of_neurons in self.hidden_layers:
            l = tf.keras.layers.Dense(number_of_neurons, self.activation)
            layer_list.append(l)

        layer_list.append(tf.keras.layers.Dense(10))
        model = tf.keras.Sequential(layer_list)
        return model

    def compile_model(self):
        self.model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

    def train_model(self, epochs=10):
        self.model.fit(self.mnist.train_images, self.mnist.train_labels, epochs=epochs)

    def evaluate_model(self, verbosity=2):
        test_loss, test_accuracy = self.model.evaluate(self.mnist.test_images, self.mnist.test_labels, verbose=verbosity)

        return test_loss, test_accuracy

    def prediction(self, pckl=False):
        prediction_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])
        pred = prediction_model.predict(self.mnist.test_images)
        if pckl == True:
            now = datetime.now()
            fname = self.mnist.name + "MLPPredictions" + now.strftime("%Y%m%d_%H%M%S") + '.pickle'
            f = open(fname, "wb")
            pickle.dump(pred, f)
            f.close()

        return pred

    def confusion_mat(self, pckl=False):
        cp = []
        predictions = self.prediction()
        for pred in predictions:
            cp.append(np.argmax(pred))

        conf_pred = np.array(cp)

        confusion = confusion_matrix(self.mnist.test_labels, conf_pred)
        loss, acc = self.evaluate_model(verbosity=0)
        if pckl == True:
            now = datetime.now()
            mname = self.mnist.name + "MLPConfusion" + now.strftime("%Y%m%d_%H%M%S") + '.pickle'
            f = open(mname, "wb")
            pickle.dump(confusion, f)
            f.close()

            op_dict = {}
            op_dict['Type'] = self.mnist.name
            op_dict['Flavour'] = 'MLP'
            op_dict['Loss'] = loss
            op_dict['Accuracy'] = acc
            op_dict['Layer code'] = self.hidden_layers
            op_dict['Labels'] = self.mnist.class_names
            op_dict['Confusion'] = confusion
            op_dict['Prediction'] = predictions
            mname = "../assets/" + self.mnist.name + "Output" + now.strftime("%Y%m%d_%H%M%S") + '.pickle'
            f = open(mname, "wb")
            pickle.dump(op_dict, f)
            f.close()

        return confusion


    def save_model(self, fname=""):
        if fname == "":
            now = datetime.now()
            fname = self.mnist.name + "MLP" + now.strftime("%Y%m%d_%H%M%S") + '.hdf5'
            self.model.save(fname)
            return fname

        if fname.split('.')[-1] != "hdf5":
            fname = fname + '.hdf5'

        self.model.save(fname)
        return fname

    def load_model(self, fname):
        if fname == "":
            print("Please provide a filename to load")
            return
        else:
            self.model = tf.keras.models.load_model(fname)



class CNN():
    def __init__(self, mnist, hidden_layers, activation):
        self.mnist = mnist
        self.num_layers = len(hidden_layers)
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.model = self.build_model()

    def build_model(self):
        t_s = self.mnist.train_images.shape
        if len(t_s) < 4:
            layer_list = [layers.Conv2D(self.hidden_layers[0][0],
                            (self.hidden_layers[0][1], self.hidden_layers[0][1]),
                            input_shape=(28, 28, 1), activation=self.activation)]
            self.mnist.train_images = self.mnist.train_images.reshape((self.mnist.train_images.shape[0], 28, 28, 1))
            self.mnist.test_images = self.mnist.test_images.reshape((self.mnist.test_images.shape[0], 28, 28, 1))
        else:
            layer_list = [layers.Conv2D(self.hidden_layers[0][0],
                            (self.hidden_layers[0][1], self.hidden_layers[0][1]),
                            input_shape=(32, 32, 3), activation=self.activation)]

        i=0
        for filter, kernel in self.hidden_layers:
            if i==0:
                i=1
                continue
            else:
                i=1
                l = layers.Conv2D(filter, (kernel, kernel), activation=self.activation)
                layer_list.append(l)

        model = tf.keras.models.Sequential(layer_list)
        model.add(layers.Flatten())
        model.add(layers.Dense(10))
        print(model.summary())
        return model

    def compile_model(self):
        self.model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

    def train_model(self, epochs=10):
        self.model.fit(self.mnist.train_images, self.mnist.train_labels, epochs=epochs)

    def evaluate_model(self, verbosity=2):
        test_loss, test_accuracy = self.model.evaluate(self.mnist.test_images, self.mnist.test_labels, verbose=verbosity)

        return test_loss, test_accuracy

    def prediction(self, pckl=False):
        prediction_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])
        pred = prediction_model.predict(self.mnist.test_images)
        if pckl == True:
            now = datetime.now()
            fname = self.mnist.name + "CNNPredictions" + now.strftime("%Y%m%d_%H%M%S") + '.pickle'
            f = open(fname, "wb")
            pickle.dump(pred, f)
            f.close()

        return pred

    def confusion_mat(self, pckl=False):
        cp = []
        predictions = self.prediction()
        for pred in predictions:
            cp.append(np.argmax(pred))

        conf_pred = np.array(cp)

        confusion = confusion_matrix(self.mnist.test_labels, conf_pred)
        loss, acc = self.evaluate_model(verbosity=0)
        if pckl == True:
            now = datetime.now()
            mname = self.mnist.name + "CNNConfusion" + now.strftime("%Y%m%d_%H%M%S") + '.pickle'
            f = open(mname, "wb")
            pickle.dump(confusion, f)
            f.close()

            op_dict = {}
            op_dict['Type'] = self.mnist.name
            op_dict['Flavour'] = 'CNN'
            op_dict['Loss'] = loss
            op_dict['Accuracy'] = acc
            op_dict['Layer code'] = self.hidden_layers
            op_dict['Labels'] = self.mnist.class_names
            op_dict['Confusion'] = confusion
            op_dict['Prediction'] = predictions
            mname = "../assets/" + self.mnist.name + "Output" + now.strftime("%Y%m%d_%H%M%S") + '.pickle'
            f = open(mname, "wb")
            pickle.dump(op_dict, f)
            f.close()

        return confusion


    def save_model(self, fname=""):
        if fname == "":
            now = datetime.now()
            fname = self.mnist.name + "CNN" + now.strftime("%Y%m%d_%H%M%S") + '.hdf5'
            self.model.save(fname)
            return fname

        if fname.split('.')[-1] != "hdf5":
            fname = fname + '.hdf5'

        self.model.save(fname)
        return fname

    def load_model(self, fname):
        if fname == "":
            print("Please provide a filename to load")
            return
        else:
            self.model = tf.keras.models.load_model(fname)

class ModCNN():
    def __init__(self, mnist, hidden_layers, activation):
        self.mnist = mnist
        self.num_layers = len(hidden_layers)
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.model = self.build_model()

    def build_model(self):

        model = models.Sequential()
        if self.mnist.name == 'cifar10':
            model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        else:
            model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
            self.mnist.train_images = self.mnist.train_images.reshape((self.mnist.train_images.shape[0], 28, 28, 1))
            self.mnist.test_images = self.mnist.test_images.reshape((self.mnist.test_images.shape[0], 28, 28, 1))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.1))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.1))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.Dropout(0.1))
        model.add(layers.Flatten())
        model.add(layers.Dense(1024, activation='relu', kernel_initializer='he_uniform'))
        model.add(layers.Dropout(0.1))
        # model.add(layers.Dense(384, activation='relu', kernel_initializer='he_uniform'))
        # model.add(layers.Dropout(0.2))
        model.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(10, activation='softmax'))
        # compile model
        # opt = tf.keras.optimizers.SGD(lr=0.001, momentum=0.9)
        print((model.summary()))
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        return model

    def compile_model(self):
        self.model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

    def train_model(self, epochs=30):
        history = self.model.fit(self.mnist.train_images, self.mnist.train_labels, epochs=epochs)
        return history

    def evaluate_model(self, verbosity=2):
        test_loss, test_accuracy = self.model.evaluate(self.mnist.test_images, self.mnist.test_labels, verbose=verbosity)

        return test_loss, test_accuracy

    def prediction(self, pckl=False):
        prediction_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])
        pred = prediction_model.predict(self.mnist.test_images)
        if pckl == True:
            now = datetime.now()
            fname = self.mnist.name + "CNNPredictions" + now.strftime("%Y%m%d_%H%M%S") + '.pickle'
            f = open(fname, "wb")
            pickle.dump(pred, f)
            f.close()

        return pred

    def confusion_mat(self, pckl=False):
        cp = []
        predictions = self.prediction()
        for pred in predictions:
            cp.append(np.argmax(pred))

        conf_pred = np.array(cp)

        confusion = confusion_matrix(self.mnist.test_labels, conf_pred)
        loss, acc = self.evaluate_model(verbosity=0)
        if pckl == True:
            now = datetime.now()
            mname = self.mnist.name + "CNNConfusion" + now.strftime("%Y%m%d_%H%M%S") + '.pickle'
            f = open(mname, "wb")
            pickle.dump(confusion, f)
            f.close()

            op_dict = {}
            op_dict['Type'] = self.mnist.name
            op_dict['Flavour'] = 'CNN'
            op_dict['Loss'] = loss
            op_dict['Accuracy'] = acc
            op_dict['Layer code'] = self.hidden_layers
            op_dict['Labels'] = self.mnist.class_names
            op_dict['Confusion'] = confusion
            op_dict['Prediction'] = predictions
            mname = "../assets/" + self.mnist.name + "Output" + now.strftime("%Y%m%d_%H%M%S") + '.pickle'
            f = open(mname, "wb")
            pickle.dump(op_dict, f)
            f.close()

        return confusion


    def save_model(self, fname=""):
        if fname == "":
            now = datetime.now()
            fname = self.mnist.name + "CNN" + now.strftime("%Y%m%d_%H%M%S") + '.hdf5'
            self.model.save(fname)
            return fname

        if fname.split('.')[-1] != "hdf5":
            fname = fname + '.hdf5'

        self.model.save(fname)
        return fname

    def load_model(self, fname):
        if fname == "":
            print("Please provide a filename to load")
            return
        else:
            self.model = tf.keras.models.load_model(fname)

class NN_layers():
    def __init__(self, fname):
        self.name = self.parse_fname(fname)
        self.model = self.load_model(fname)
        self.conv_layers = self.get_conv_layers()


    def parse_fname(self, fname):
        if 'cifar10' in fname:
            return 'cifar10'
        elif 'fashion' in fname:
            return 'fashion'
        else:
            return 'numbers'

    def load_model(self, fname):
        print(fname)
        return models.load_model(fname)

    def show_conv_layers(self):
        for layer in self.model.layers:
            if 'conv' not in layer.name:
                continue
            filters, biases = layer.get_weights()
            print(layer.name, filters.shape)

    def get_conv_layers(self):
        l_dict = {}
        i=0
        for layer in self.model.layers:
            if 'conv' not in layer.name:
                continue
            filters, biases = layer.get_weights()
            fmin, fmax = filters.min(), filters.max()
            filters = (filters-fmin)/(fmax-fmin)
            l_dict[i] = filters
            i+=1

        return l_dict










