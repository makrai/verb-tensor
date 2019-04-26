#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import defaultdict
import glob
from itertools import groupby
import operator
import os
import re

import numpy as np
import pandas as pd
import pickle
import random
import sparse
#import tensorly as tl
#import tensorly.decomposition as decomp
import sktensor
import urllib3
#import wget

import matplotlib.pyplot as plt
#get_ipython().run_line_magic('pylab', 'inline')
#pylab.rcParams['figure.figsize'] = (10, 6)

import logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(msecs)d %(levelname)-8s [%(lineno)d] %(message)s')

if not os.path.exists('cp_orth.py'):
    wget.download('http://web.stanford.edu/~vsharan/cp_orth.py')
from cp_orth import orth_als


# In[2]:


projdir = '/mnt/permanent/home/makrai/project/verb-tensor'


# In[3]:


def mazsola_reader():
    pickle_path = os.path.join(projdir, 'mazsola.pkl')
    if os.path.exists(pickle_path):
        logging.info('Loading mazsola dict from {}'.format(pickle_path))
        return pickle.load(open(pickle_path, mode='rb'))
    logging.info('Reading mazsola...'.format(pickle_path))
    path = '/mnt/permanent/Language/Hungarian/Dic/sass15-535k-igei-szerkezet/mazsola_adatbazis.txt'
    occurrence = defaultdict(int)#lambda: defaultdict(lambda: defaultdict(int)))
    margianls = [defaultdict(int) for _ in range(3)]
    with open(path) as infile:
        for i, line in enumerate(infile):
            if not i % 500000:
                logging.info('{:.0%}'.format(i/27970403))
            record = defaultdict(str)
            for token in line.strip().split():
                case_stem = re.split('@@', token)
                if len(case_stem) == 1:
                    continue
                try:
                    case, stem = case_stem
                except:
                    logging.warning(line.strip())
                record[case] = stem
            occurrence[record['NOM'], record['stem'], record['ACC']] += 1
            for i, mode in enumerate(['NOM', 'stem', 'ACC']):
                margianls[i][record[mode]] += 1            
    result = occurrence, margianls
    pickle.dump(result, open(pickle_path, mode='wb'))
    return result                


# In[4]:


def get_tensor(middle_end='sktensor', cutoff=10):
    logging.info('Reweighting: log')
    verb_tensor_path = os.path.join(projdir, '{}/tensor_{}.pkl'.format(middle_end, cutoff))
    if os.path.exists(verb_tensor_path):
        logging.info('Loading tensor from {}'.format(verb_tensor_path))
        tensor, indices = pickle.load(open(verb_tensor_path, mode='rb'))
        logging.debug(tensor.shape)
        return tensor, indices
    occurrence, marginals = mazsola_reader()
    def get_index(freq_dict):
        items = sorted(filter(lambda item: item[1] >= cutoff, freq_dict.items()), key=operator.itemgetter(1), 
                       reverse=True)
        logging.debug(items[-3:])
        return dict([(w, i) for i, (w, f) in enumerate(items)])

    coords, data = ([], [], []), []
    indices = [get_index(fd) for fd in marginals]
    logging.info('Building tensor...')
    logging.info('  Pupulating lists...')
    for i, ((svo), freq) in enumerate(occurrence.items()):
        if not i % 2000000:
            logging.debug('    {:,}'.format(i))#'{} {}'.format(svo[1], freq))
        for i, word in enumerate(svo):
            if svo[i] not in indices[i]:
                break
        else:
            for i, word in enumerate(svo):
                coords[i].append(indices[i][svo[i]])
            data.append(np.log(freq))
    logging.info('  Creating array')
    shape = tuple(map(len, indices))
    logging.info(shape)
    if middle_end == 'tensorly':
        tensor = sparse.COO(coords, data, shape=shape)#, has_duplicates=False)
    elif middle_end == 'sktensor':
        tensor = sktensor.sptensor(coords, data, shape=shape)
    else:
        raise NotImplementedError
    pickle.dump((tensor, indices), open(verb_tensor_path, mode='wb'))
    logging.info(tensor)
    return tensor, indices


# In[ ]:


def decomp(cutoff, dim):
    logging.info((cutoff, dim))
    filen_base = os.path.join(projdir, 'sktensor/decomp_{}_{}'.format(cutoff, dim))
    if os.path.isfile('{}.{}'.format(filen_base, 'pkl')) or os.path.isfile('{}.{}'.format(filen_base, 'err')):
        logging.info('File exists {}'.format(glob.glob(filen_base+'.*')))
        return
    vtensor, indices = get_tensor(cutoff=cutoff)
    try:
        result = orth_als(vtensor, dim)
        pickle.dump(result, open('{}.{}'.format(filen_base, 'pkl'), mode='wb'))
    except Exception as e:
        with open('{}.{}'.format(filen_base, 'err'), mode='w') as logfile:
            logging.error(e)
            logfile.write(e)


# In[ ]:


def show_expers(feature='exectimes'):
    tabular = []
    mx = []
    for filen in glob.glob(os.path.join(projdir, 'sktensor/decomp_*.pkl')):
        logging.debug('')
        _, cutoff, dim = os.path.splitext(filen)[0].split('_')
        cutoff, dim = map(int, (cutoff, dim))
        ktensor, fit, n_iterations, exectimes = pickle.load(open(filen, mode='rb'))
        tabular.append((cutoff, dim, ktensor.shape))
        mx.append([cutoff, dim, sum(exectimes)/60])
    mx = np.array(mx)
    print(sorted(tabular))
    plt.scatter(np.array(mx).T[0], mx.T[1], c=mx.T[2])
    plt.colorbar()
    plt.xscale('log')

show_expers()
# In[ ]:


def rand_elem(list1):
    return list1[np.random.randint(0, len(list1))]


# In[ ]:


for exp in range(13, -1, -1):
    decomp(2**exp, 100)

