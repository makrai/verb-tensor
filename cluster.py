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
                 min_cluster_size=5, min_samples=None, metric='cosine'):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        tensor_dir = '/mnt/store/home/makrai/project/verb-tensor/depCC/'
        logging.info('Loading tensor and index..')
        self.ktensor, _, _, _ = pickle.load(open(os.path.join(
            tensor_dir, 
            'ktensor_{}_{}_{}.pkl'.format(weight, cutoff, rank)), mode='rb'))
        logging.debug(self.ktensor.shape)
        _, self.index = pickle.load(open(os.path.join(
            tensor_dir, 
            'sparstensr_{}_{}.pkl'.format(weight, cutoff)), mode='rb'))
        logging.info('UMAP for visualization..')
        self.embed_visu = self.do_umap(2)
        logging.info('UMAP for clustering..') 
        self.embed_clus = self.do_umap(clusser_dim)

    def do_umap(self, n_components):
        if self.ktensor.U[1].shape[1] <= n_components:
            return self.ktensor.U[1]
        mapping = umap.UMAP(n_components=n_components, metric=self.metric)
        return mapping.fit_transform(self.ktensor.U[1])
    
    def do_cluster(self):
        logging.info('Clustering..')
        clusser = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size,
                                  min_samples=self.min_samples)
        # min_samples defaults to whatever min_cluster_size is set to
        logging.debug('')
        self.labels = clusser.fit_predict(self.embed_clus)
        plt.scatter(self.embed_visu.T[0], self.embed_visu.T[1], c=self.labels, s=2)
        n_clusters = len(set(self.labels))
        if n_clusters < 100: 
            boundaries = np.arange(n_clusters+1)/(n_clusters+1) 
        else:
            boundaries = None
        plt.colorbar(ticks=boundaries, boundaries=boundaries)                   

    def write_cluters(self, n_clust_show=30, show_size_of=100,
                      sort_sizes='descending'):
        logging.info('Writing clusters..')
        assert sort_sizes in ['ascending', 'descending', 'rand']
        clus_len_l = [(key, len(list(group))) 
                      for key, group in groupby(sorted(self.labels))]
        if sort_sizes is not 'rand':
            clus_len_l = sorted(clus_len_l, key=operator.itemgetter(1),
                                reverse=sort_sizes=='descending')
        clusters, lens = zip(*list(clus_len_l))
        n_clusters = len(clus_len_l)
        header = '{} clusters, sizes: {}{}'.format(
            n_clusters, ' '.join(map(str, lens[:show_size_of])), 
            '..' if n_clusters>show_size_of else '')
        print(header)
        verb_name = 'ROOT' # 'stem'
        logging.debug('All verbs: {}'.format(
            ' '.join(self.index[verb_name].inverse[i] for i in range(10))))
        for cluster_i, clus_len in list(clus_len_l)[:n_clust_show+1]:
            print((cluster_i, clus_len, 
                   [self.index[verb_name].inverse[i] 
                    for i in sorted(np.where(self.labels==cluster_i)[0])[:9]]))
        print('\n{}'.format(header))
    
    def main(self):
        self.do_cluster()
        self.write_cluters()

if __name__ == '__main__': 
    ClusterVerbs().main()

