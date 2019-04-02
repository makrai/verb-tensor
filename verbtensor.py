#!/usr/bin/env python
# coding: utf-8


import argparse
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

import logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(levelname)-8s [%(lineno)d] %(message)s')

if not os.path.exists('cp_orth.py'):
    wget.download('http://web.stanford.edu/~vsharan/cp_orth.py')
from cp_orth import orth_als


class keydefault_dict(dict):
    def __missing__(self, key):
        return ('', key)

class VerbTensorDecomposition():
    def __init__(self, separate_prev, cutoff, rank):
        self.separate_prev = separate_prev
        self.cutoff = cutoff
        self.rank = rank
        self.projdir = os.path.join(
            '/mnt/store/home/makrai/project/verb-tensor/smooth', 
            'prev_sep' if self.separate_prev else 'holis')

    def get_prev_verb(self):
        names = ['lemma', 'token_freq', 'pos', 'doc_freq', 'normalized']
        prevlex = pd.read_csv(
                '/home/makrai/repo/prevlex/PrevLex.txt', sep='\t', header=None,
                names=names)
        #prevlex.head()
        prev_verb = keydefault_dict()
        for prev_plus_verb in prevlex.lemma:
            prev, verb = prev_plus_verb.split('+')
            prev_verb[prev+verb] = (prev, verb)
        return prev_verb

    def mazsola_reader(self):
        pickle_path = os.path.join(self.projdir, 'mazsola.pkl')
        if os.path.exists(pickle_path):
            logging.info('Loading mazsola dict from {}'.format(pickle_path))
            return pickle.load(open(pickle_path, mode='rb'))
        logging.info('Reading mazsola...'.format(pickle_path))
        path = '/mnt/permanent/Language/Hungarian/Dic/sass15-535k-igei-szerkezet/mazsola_adatbazis.txt'
        if self.separate_prev:
            prev_verb = get_prev_verb()
        occurrence = defaultdict(int)
        margianls = [defaultdict(int) for _ in range(3 + bool(self.separate_prev))]
        verb_name_l =  ['prev', 'verb'] if self.separate_prev else ['stem']
        ax_names = ['NOM'] +  verb_name_l + ['ACC']
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

                if self.separate_prev: 
                    record['prev'], record['verb'] = prev_verb[record['stem']]
                occurrence[tuple(record[name] for name in ax_names)] += 1
                for i, mode in enumerate(ax_names):
                    margianls[i][record[mode]] += 1            
        result = occurrence, margianls
        pickle.dump(result, open(pickle_path, mode='wb'))
        return result                

    def get_tensor(self):
        logging.info('Reweighting: log')
        verb_tensor_path = os.path.join(self.projdir, 'tensor_{}.pkl').format(self.cutoff)
        if os.path.exists(verb_tensor_path):
            logging.info('Loading tensor from {}'.format(verb_tensor_path))
            tensor, indices = pickle.load(open(verb_tensor_path, mode='rb'))
            logging.debug(tensor.shape)
            return tensor, indices
        occurrence, marginals = self.mazsola_reader()
        def get_index(freq_dict):
            items = sorted( 
                    filter(lambda item: item[1] >= self.cutoff, freq_dict.items()),
                    key=operator.itemgetter(1), reverse=True)
            logging.debug(items[-3:])
            return dict([(w, i) for i, (w, f) in enumerate(items)])

        coords, data = tuple([] for _ in range(3 + bool(self.separate_prev))), []
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
                data.append(np.log(freq + 1)
        logging.info('  Creating array')
        shape = tuple(map(len, indices))
        logging.info(shape)
        # if middle_end == 'tensorly':
        # tensor = sparse.COO(coords, data, shape=shape)#, has_duplicates=False)
        tensor = sktensor.sptensor(coords, data, shape=shape)
        pickle.dump((tensor, indices), open(verb_tensor_path, mode='wb'))
        logging.info(tensor)
        return tensor, indices

    def decomp(self):
        logging.info((self.separate_prev, self.cutoff, self.rank))
        filen_base = os.path.join(self.projdir, 'decomp_{}_{}').format(self.cutoff, self.rank)
        if os.path.isfile('{}.{}'.format(filen_base, 'pkl')):
            logging.info('File exists {} {}'.format(self.cutoff, self.rank))
            return
        vtensor, indices = self.get_tensor()
        try:
            result = orth_als(vtensor, self.rank)
            pickle.dump(result, open('{}.{}'.format(filen_base, 'pkl'), mode='wb'))
        except Exception as e:
            with open('{}.{}'.format(filen_base, 'err'), mode='w') as logfile:
                logfile.write('{}'.format(e))
            logging.exception(e) 


def parse_args(): 
    parser = argparse.ArgumentParser(
        description='Decompose a tensor of verb and argument cooccurrences') 
    parser.add_argument('--separate_prev', action='store_true')
    parser.add_argument('--cutoff', type=int, default=2**10)
    parser.add_argument('--rank', type=int, default=50)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    decomposer = VerbTensorDecomposition(
            args.separate_prev, args.cutoff, args.rank).decomp() 
