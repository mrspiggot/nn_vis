import sklearn
import numpy as np
import pandas as pd
from numpy import linalg
from scipy.spatial.distance import squareform, pdist
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold._t_sne import (_joint_probabilities,
                                    _kl_divergence)
import pickle
from util_classes.MNIST import NN_layers, MNISTBase

RS = 20150101

class DisplayTSNE():
    def __init__(self, name='sk_digits', perplexity=30, early_exaggeration=12.0, dimensions=2):
        self.source = name
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.dimensions = dimensions
        self.anim_df = pd.DataFrame()
        self.X, self.y = self.get_data()
        self.static_points = self.static_data_chart()
        # Pairwise distances between all data points.
        self.D = pairwise_distances(self.X, squared=True)
        # Similarity with constant sigma.
        self.P_constant = self._joint_probabilities_constant_sigma(self.D, .002)
        # Similarity with variable sigma.
        self.P_binary = _joint_probabilities(self.D, 30., False)
        # The output of this function needs to be reshaped to a square matrix.
        self.P_binary_s = squareform(self.P_binary)
        self.positions = []
        sklearn.manifold._t_sne._gradient_descent = self._gradient_descent
        self.animated_points, self.anim_df = self.get_animation()


    def get_data(self):
        if self.source == 'sk_digits':
            digits = load_digits()
            X = np.vstack([digits.data[digits.target == i]
                           for i in range(10)])
            y = np.hstack([digits.target[digits.target == i]
                           for i in range(10)])
        elif self.source == 'fashion':
            nn = NN_layers("fashionCNN20211108_095455.hdf5")
            mn = MNISTBase(nn.name)

            X = nn.dense_layers(mn.test_images)
            y = mn.test_labels
        elif self.source == 'cifar10':
            nn = NN_layers("cifar10CNN20211108_004852.hdf5")
            mn = MNISTBase(nn.name)

            X = nn.dense_layers(mn.test_images)
            y = mn.test_labels

        else:
            nn = NN_layers("numbersCNN20211108_101910.hdf5")
            mn = MNISTBase(nn.name)

            X = nn.dense_layers(mn.test_images)
            y = mn.test_labels

        X = X[:3500,:]
        y = y[:3500]
        print("X", X.shape)
        print("y", y.shape)
        return X, y

    def static_data_chart(self):
        return TSNE(random_state=RS, n_components=self.dimensions, perplexity=self.perplexity,
                    early_exaggeration=self.early_exaggeration).fit_transform(self.X)

    def _joint_probabilities_constant_sigma(self, D, sigma):
        P = np.exp(-D ** 2 / 2 * sigma ** 2)
        P /= np.sum(P, axis=1)
        return P

    def _gradient_descent(self, objective, p0, it, n_iter, objective_error=None, n_iter_check=1, n_iter_without_progress=30,
                          momentum=0.5, learning_rate=1000.0, min_gain=0.01,
                          min_grad_norm=1e-7, min_error_diff=1e-7, verbose=0,
                          args=[], kwargs=None):
        p = p0.copy().ravel()
        update = np.zeros_like(p)
        gains = np.ones_like(p)
        error = np.finfo(np.float).max
        best_error = np.finfo(np.float).max
        best_iter = 0

        for i in range(it, n_iter):
            # We save the current position.
            self.positions.append(p.copy())

            new_error, grad = objective(p, *args)
            error_diff = np.abs(new_error - error)
            error = new_error
            grad_norm = linalg.norm(grad)

            if error < best_error:
                best_error = error
                best_iter = i
            elif i - best_iter > n_iter_without_progress:
                break
            if min_grad_norm >= grad_norm:
                break
            if min_error_diff >= error_diff:
                break

            inc = update * grad >= 0.0
            dec = np.invert(inc)
            gains[inc] += 0.05
            gains[dec] *= 0.95
            np.clip(gains, min_gain, np.inf)
            grad *= gains
            update = momentum * update - learning_rate * grad
            p += update

        return p, error, i

    def get_animation(self):
        X_proj = TSNE(random_state=RS, n_components=self.dimensions, perplexity=self.perplexity,
              early_exaggeration=self.early_exaggeration,
              learning_rate=500,
              n_iter_without_progress=100000).fit_transform(self.X)


        X_iter = np.dstack(position.reshape(-1, self.dimensions)
                           for position in self.positions)

        index = pd.MultiIndex.from_product([range(s) for s in X_iter.shape])
        mdim = pd.DataFrame({'X_iter': X_iter.flatten()}, index=index)['X_iter']
        mdim = mdim.unstack().swaplevel().sort_index()
        mdim.index.names = ['co-ord', 'i']

        df2 = mdim.stack().unstack(level=0)

        if self.source=='fashion':
            class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
            names = []
            for i in self.y:
                names.append(class_names[i])

            label_df = pd.DataFrame(data=names)
        elif self.source=='cifar10':
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
            names = []
            print("In Cifar", type(self.y), self.y.shape)
            for i in self.y.reshape(-1):
                names.append(class_names[i])

            label_df = pd.DataFrame(data=names)

        else:
            label_df = pd.DataFrame(data=self.y)
        label_df.columns = ['label']


        firsts = df2.index.get_level_values('i')
        df2['label'] = label_df.loc[firsts].values

        df2.index.names = ['image', 'iteration']
        if self.dimensions == 2:
            df2.columns = ['x', 'y', 'labels']
        elif self.dimensions == 3:
            df2.columns = ['x', 'y', 'z', 'labels']
        elif self.dimensions == 1:
            df2.columns = ['y', 'labels']
        else:
            print('"Dimensions" must be 2 or 3')

        return X_proj, df2

    def pickle_data(self, fname="tsne"):
        p_dict = {}
        p_dict['perplexity'] = self.perplexity
        p_dict['early_exaggeration'] = self.early_exaggeration
        p_dict['dimensions'] = self.dimensions
        p_dict['anim_df'] = self.anim_df

        fn = fname + "_p_" + str(self.perplexity) + "_e_" + str(self.early_exaggeration) + "_d_" + str(self.dimensions) + ".pickle"
        with open(fn, 'wb') as pkl:
            pickle.dump(p_dict, pkl, protocol=pickle.HIGHEST_PROTOCOL)

