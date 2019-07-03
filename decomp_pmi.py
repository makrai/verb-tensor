#!/usr/bin/env python
# coding: utf-8


import argparse
from bidict import bidict
from collections import defaultdict
import itertools
import os
import pandas as pd
import pickle
import lzma

from conllu import parse, parse_incr
import numpy as np
from cp_orth import orth_als
import sktensor

import logging
logging.basicConfig(level=logging.DEBUG,
        format='%(levelname)-8s [%(lineno)d] %(message)s')

class VerbTensor():
    def __init__(self, cutoff=1):#rank
        self.cutoff = cutoff
        #self.rank = rank
        self.project_dir = '/mnt/store/home/makrai/project/verb-tensor/'
        self.subproject_dir = os.path.join(self.project_dir, 'depCC/')
        self.pmi_df_filen = os.path.join(self.project_dir, 'depCC.tsv')
        self.modes = ['nsubj, ROOT, dobj']
        # mazsola: ['NOM', 'stem', 'ACC']

    def append_pmi(self, svo_count, debug_index=None):
        logging.info('Computing marginals..')
        marginal = {mode: svo_count.groupby(mode).sum() for mode in self.modes}
        logging.info('Computing 2-marginals..')
        marginal2 = {mode_pair: svo_count.groupby(list(mode_pair)).sum()
                 for mode_pair in itertools.combinations(self.modes, 2)}
        log_total = np.log2(svo_count.freq.sum())
        for mode in self.modes:
            svo_count = svo_count.join(marginal[mode], on=mode,
                    rsuffix='_{}'.format(mode))
        logging.info('Computing Dice..')
        for mode_pair in itertools.combinations(self.modes, 2):
            svo_count = svo_count.join(marginal2[mode_pair], on=mode_pair,
                    rsuffix='_{}'.format(mode_pair))
        svo_count['dice'] = 3*svo_count.freq # This is only the numerator.
        svo_count['freq2'] = svo_count.freq
        if debug_index:
            logging.debug(svo_count.loc[debug_index])
        svo_count['dice_denom'] = 0
        for mode in self.modes:
            svo_count.dice_denom += svo_count['freq_{}'.format(mode)]
            if debug_index:
                logging.debug(svo_count.loc[debug_index])
        svo_count.dice /= svo_count.dice_denom
        del svo_count['dice_denom']
        if debug_index:
            logging.debug(svo_count.loc[debug_index])
        logging.info('Computing PMI variants..')
        for name in svo_count.columns[4:]:
            if name != 'dice':
                svo_count[name] = np.log2(svo_count[name]) - log_total
        svo_count['pmi'] = svo_count.freq2
        svo_count['iact_info'] = -svo_count.freq2
        if debug_index:
            logging.debug(svo_count.loc[debug_index])
        for mode in self.modes:
            svo_count.pmi -= svo_count['freq_{}'.format(mode)]
            svo_count.iact_info += svo_count['freq_{}'.format(mode)]
            if debug_index:
                logging.debug(svo_count.loc[debug_index])
        for mode_pair in itertools.combinations(self.modes, 2):
            svo_count.iact_info -= svo_count['freq_{}'.format(mode_pair)]
            if debug_index:
                logging.debug(svo_count.loc[debug_index])
        svo_count.pmi = np.max(svo_count.pmi, 0)
        svo_count.iact_info = np.max(svo_count.iact_info, 0) # TODO Is it OK?
        logging.info('Computing salience..')
        svo_count['salience'] = svo_count.pmi * svo_count.freq2
        svo_count['iact_sali'] = svo_count.iact_info * svo_count.freq2
        logging.info('Writing to {}..'.format(self.pmi_df_filen))
        svo_count.to_csv(self.pmi_df_filen, sep='\t', index=False,
                         float_format='%.5g')
        return svo_count, log_total

    def create_sparse(self, df, weight):
        logging.debug(weight)
        if weight == 'freq': # TODO Is this needed?
            df[weight] = np.log(df[weight] + 1)
        self.index = {
            mode: bidict(df.groupby(mode)[weight].sum().argsort().to_dict())
            for mode in self.modes}
        for mode in self.modes:
            self.index[mode][np.nan] = len(self.index[mode])
            df['{}_i'.format(mode)] = df[mode].apply(self.index[mode].get)
        logging.debug('Creating tensor..')
        coords = df[['{}_i'.format(mode)
                     for mode in self.modes]].T.to_records(index=False)
        coords = tuple(map(list, coords))
        data = df[weight].values
        shape=tuple(len(self.index[mode]) for mode in self.modes)
        logging.debug(([len(y) for y in coords], len(data)))
        logging.info(shape)
        self.sparse_tensor = sktensor.sptensor(coords, data, shape=shape)
        pickle.dump((self.sparse_tensor, self.index), open(os.path.join(
            self.subproject_dir, self.sparse_filen), mode='wb'))

    def get_sparse(self, weight):
        if os.path.exists(self.sparse_filen):
            logging.info('Loading tensor..')
            self.sparse_tensor, self.index =  pickle.load(open(
                self.sparse_filen, mode='rb'))
        else:
            logging.info(
                'Reading association weights from {}..'.format(
                    self.pmi_df_filen))
            self.pmi_df = pd.read_csv(
                self.pmi_df_filen, sep='\t', keep_default_na=False)
            self.create_sparse(
                self.pmi_df[self.pmi_df.freq>self.cutoff].copy(), # TODO ?
                weight)

    def decomp(self, weight, rank):
        logging.info((weight, rank, self.cutoff))
        decomp_filen = os.path.join(self.subproject_dir, '{}_{}_{}_{}.pkl').format(
            'ktensor', weight, self.cutoff, rank)
        if os.path.exists(decomp_filen):
            logging.warning('File exists')
            return
        self.sparse_filen = os.path.join(self.subproject_dir, '{}_{}_{}.pkl'.format(
            'sparstensr', weight, self.cutoff))
        self.get_sparse(weight)
        result = orth_als(self.sparse_tensor, rank)
        pickle.dump(result, open(decomp_filen, mode='wb'))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Decompose a tensor of verb and argument cooccurrences')
    parser.add_argument('--weight', default='freq')
    parser.add_argument('--cutoff', type=int, default=0) # TODO ?
    parser.add_argument('--rank', type=int, default=100)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    decomposer = VerbTensor(cutoff=args.cutoff)
    #for weight in ['freq', 'pmi', 'iact_info', 'salience', 'iact_sali', 'dice']:
    #try:
    decomposer.decomp(args.weight, args.rank)
    #except Exception as e:
    #logging.warning(e)
