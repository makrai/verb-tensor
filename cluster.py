# coding: utf-8

from collections import defaultdict
from itertools import groupby
import operator
import os
import re

import hdbscan
import numpy as np
import sklearn
import pickle
import random
import sparse
from sklearn.manifold import TSNE 
import sktensor
import umap
import urllib3
import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(levelname)-8s [%(lineno)d] %(message)s')

if not os.path.exists('cp_orth.py'):
    wget.download('http://web.stanford.edu/~vsharan/cp_orth.py')
from cp_orth import orth_als


class ClusterVerbs():
    def __init__(self, weight='freq', cutoff=2, rank=32, clusser_dim=10,
                 min_cluster_size=5, min_samples=None, metric='cosine',
                 init='spectral'):
        self.clusser_dim = clusser_dim
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.init = init
        tensor_dir = config['DEFAULT']['ProjectDirectory']+'tensor/0'
        logging.info('Loading tensor and index..')
        self.ktensor, _, _, _ = pickle.load(open(os.path.join(
            tensor_dir,
            'ktensor_{}_{}_{}.pkl'.format(weight, cutoff, rank)), mode='rb'))
        logging.debug(self.ktensor.shape)
        _, self.index = pickle.load(open(os.path.join(
            tensor_dir,
            'sparstensr_{}_{}.pkl'.format(weight, cutoff)), mode='rb'))

    def do_umap(self, n_components):
        if self.ktensor.U[1].shape[1] <= n_components:
            return self.ktensor.U[1]
        #mapping = umap.UMAP(n_components=n_components, metric=self.metric,
        mapping = TSNE(n_components=n_components, metric=self.metric,
                       init=self.init, 
                       method='barnes_hut' if n_components > 4 else 'exact')
        return mapping.fit_transform(self.ktensor.U[1])

    def plot_cluster(self):
        plt.scatter(self.embed_visu.T[0], self.embed_visu.T[1], c=self.labels,
                    s=2)
        n_clusters = len(set(self.labels))
        if n_clusters < 100:
            boundaries = np.arange(n_clusters+1)/(n_clusters+1)
        else:
            boundaries = None
        plt.colorbar(ticks=boundaries, boundaries=boundaries)

    def main(self):
        logging.info('Mapping for visualization..')
        self.embed_visu = self.do_umap(2)
        logging.info('Mapping for clustering..')
        self.embed_clus = self.do_umap(self.clusser_dim)
        logging.info('HDBSCAN..')
        clusser = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size,
                                  min_samples=self.min_samples)
        # min_samples defaults to whatever min_cluster_size is set to
        logging.debug('')
        self.labels = clusser.fit_predict(self.embed_clus)
        return self.index, self.labels


if __name__ == '__main__':
    ClusterVerbs().main()
