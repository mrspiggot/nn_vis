import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model

# Helper libraries
import numpy as np
from numpy import expand_dims
from datetime import datetime
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()

import base64

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
        self.history = None


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
        t_s = self.mnist.train_images.shape
        if len(t_s) > 3:
            sh = self.mnist.train_images.shape
            X, y = shuffle(self.mnist.train_images.reshape(sh[0], sh[1],  sh[2]), self.mnist.train_labels.reshape(-1))
            x_val = X[-5000:]
            y_val = y[-5000:]
            x_train = X[:-5000]
            y_train = y[:-5000]

        else:
            print("X=", self.mnist.train_images.shape)
            print("y=", self.mnist.train_labels.shape)

            X, y = shuffle(self.mnist.train_images, self.mnist.train_labels)
            x_val = X[-5000:]
            y_val = y[-5000:]
            x_train = X[:-5000]
            y_train = y[:-5000]
        history = self.model.fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val))
        self.history = history
        return history

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

    def confusion_mat(self, pckl=False, fname=""):
        cp = []
        print("fname ", fname)

        predictions = self.prediction()
        for pred in predictions:
            cp.append(np.argmax(pred))

        conf_pred = np.array(cp)

        confusion = confusion_matrix(self.mnist.test_labels, conf_pred)
        test_acc = confusion.trace() / confusion.sum()
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
            op_dict['Historic Loss'] = self.history.history['loss']
            op_dict['Historic Accuracy'] = self.history.history['accuracy']
            op_dict['Validation Loss'] = self.history.history['val_loss']
            op_dict['Validation Accuracy'] = self.history.history['val_accuracy']
            op_dict['Test Accuracy'] = test_acc
            if fname == "":

                layer_list = ""
                for neurons in self.hidden_layers:
                    layer_list += (str(neurons)+"_")

                fname = "../assets/" + self.mnist.name + "Output" + layer_list + now.strftime("%Y%m%d_%H%M%S") + "_" + str(test_acc) + '.pickle'
            else:
                fname = "assets/" + fname

            print("fname2 ", fname)
            f = open(fname, "wb")
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
        self.history = None

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
        X, y = shuffle(self.mnist.train_images, self.mnist.train_labels)
        x_val = X[-5000:]
        y_val = y[-5000:]
        x_train = X[:-5000]
        y_train = y[:-5000]
        history = self.model.fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val))
        self.history = history
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
        test_acc = confusion.trace() / confusion.sum()
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
            op_dict['Historic Loss'] = self.history.history['loss']
            op_dict['Historic Accuracy'] = self.history.history['accuracy']
            op_dict['Validation Loss'] = self.history.history['val_loss']
            op_dict['Validation Accuracy'] = self.history.history['val_accuracy']
            op_dict['Test Accuracy'] = test_acc
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
        self.history = None

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
        X, y = shuffle(self.mnist.train_images, self.mnist.train_labels)
        x_val = X[-5000:]
        y_val = y[-5000:]
        x_train = X[:-5000]
        y_train = y[:-5000]
        history = self.model.fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val))
        self.history = history
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
        test_acc = confusion.trace() / confusion.sum()
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
            op_dict['Historic Loss'] = self.history.history['loss']
            op_dict['Historic Accuracy'] = self.history.history['accuracy']
            op_dict['Validation Loss'] = self.history.history['val_loss']
            op_dict['Validation Accuracy'] = self.history.history['val_accuracy']
            op_dict['Test Accuracy'] = test_acc
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
        self.img = None
        self.testImgArr = None


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

    def load_image(self, image):
        self.img = load_img(image, target_size=(28,28))
        ti = img_to_array(self.img)
        self.testImgArr = np.dot(ti[...,:3], [0.2989, 0.5870, 0.1140]).astype(int)
        print("Loading ", image)

    def display_conv1(self):
        cnn_model = Model(inputs=self.model.inputs, outputs=self.model.layers[0].output)
        t_img = expand_dims(self.testImgArr, axis=0)
        activations = cnn_model.predict(t_img)
        return activations

    def display_3_convs(self):
        ixs = [0, 3, 6]
        outputs = [self.model.layers[i].output for i in ixs]
        model = Model(inputs=self.model.inputs, outputs=outputs)
        t_img = expand_dims(self.testImgArr, axis=0)
        feature_maps = model.predict(t_img)
        for fmap in feature_maps:
            print(type(fmap), len(fmap), fmap.shape)

    def dense_layers(self, y):
        for i in range(len(self.model.layers)):
            layer = self.model.layers[i]
            # check for convolutional layer
            if 'dense' not in layer.name:
                continue
            # summarize output shape
            print(i, layer.name, layer.output.shape)

        tsne_model = Model(inputs=self.model.inputs, outputs=self.model.layers[11].output)
        tsne_activations = tsne_model.predict(y)

        return tsne_activations


class Ensemble():
    def __init__(self, data_set, layer_list, activation='relu', epochs=30):
        self.data_set = MNISTBase(data_set)
        self.af = activation
        self.ep = epochs
        self.model_list = self.build_models(layer_list)
        self.ec = self.ensemble_confusion()

    def build_models(self, layer_list):
        ml = []
        for layers in layer_list:
            mn = MNISTBase(self.data_set.name)
            mod = MLP(mn, layers, self.af)
            mod.compile_model()
            mod.train_model(self.ep)
            mod.evaluate_model(1)
            mod.confusion_mat(True)
            mod.save_model()
            ml.append(mod)

        return ml

    def ensemble_confusion(self):

        if len(self.data_set.test_images.shape) > 3:
            ti = tf.image.rgb_to_grayscale(self.data_set.test_images)
            pred = [tf.keras.Sequential([imodel.model, tf.keras.layers.Softmax()]).predict(ti) for imodel in self.model_list]
        else:
            pred = [tf.keras.Sequential([imodel.model, tf.keras.layers.Softmax()]).predict(self.data_set.test_images) for imodel in self.model_list]

        pred = np.array(pred)
        ensemble_prediction = np.sum(pred, axis=0)
        ep = np.argmax(ensemble_prediction, axis=1)
        confusion = confusion_matrix(self.data_set.test_labels, ep)
        print(confusion, confusion.trace() / confusion.sum())
        return confusion



















