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

RS = 20150101
PERPLEXITY = 30
EARLY_EXAGGERATION = 30.0

class DisplayTSNE():
    def __init__(self, name='sk_digits'):
        self.source = name
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

        return X, y

    def static_data_chart(self):
        return TSNE(random_state=RS, perplexity=PERPLEXITY, early_exaggeration=EARLY_EXAGGERATION).fit_transform(self.X)

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
        X_proj = TSNE(random_state=RS, perplexity=PERPLEXITY,
              early_exaggeration=EARLY_EXAGGERATION,
              learning_rate=500,
              n_iter_without_progress=100000).fit_transform(self.X)

        X_iter = np.dstack(position.reshape(-1, 2)
                           for position in self.positions)

        index = pd.MultiIndex.from_product([range(s) for s in X_iter.shape])
        mdim = pd.DataFrame({'X_iter': X_iter.flatten()}, index=index)['X_iter']
        mdim = mdim.unstack().swaplevel().sort_index()
        mdim.index.names = ['co-ord', 'i']


        label_df = pd.DataFrame(data=self.y)
        label_df.columns = ['label']


        firsts = mdim.index.get_level_values('i')
        mdim['label'] = label_df.loc[firsts].values

        return X_proj, mdim

ts = DisplayTSNE()
print(ts.anim_df, ts.anim_df.info())