#!/usr/bin/env python
# coding: utf-8


import argparse
from bidict import bidict
from collections import defaultdict
import itertools
import os
import pandas as pd
import pickle
import logging
import lzma

import numpy as np
from cp_orth import orth_als
import sktensor


class VerbTensor():
    def __init__(self, input_part):
        self.part = input_part
        self.project_dir = '/mnt/permanent/home/makrai/project/verb-tensor/top_level'
        self.tensor_dir = os.path.join(self.project_dir, 'tensor')
        self.assoc_df_filen_patt = os.path.join(self.project_dir,
                                                'dataframe/assoc{}.{}')
        self.modes = ['nsubj', 'ROOT', 'dobj']
        # mazsola: ['NOM', 'stem', 'ACC']

    def append_pmi(self, write_tsv=False):
        filen = os.path.join(self.project_dir, 'dataframe/freq{}.pkl'.format(
            self.part))
        logging.info('Reading freqs from {}'.format(filen))
        svo_count = pd.read_pickle(filen)
        logging.info('Computing marginals..')
        marginal = {mode: svo_count.groupby(mode).sum() for mode in self.modes}
        marginal2 = {mode_pair: svo_count.groupby(list(mode_pair)).sum()
                     for mode_pair in itertools.combinations(self.modes, 2)}
        for mode in self.modes:
            svo_count = svo_count.join(marginal[mode], on=mode, 
                                       rsuffix='_{}'.format(mode))
        for mode_pair in itertools.combinations(self.modes, 2):
            #logging.debug(mode_pair)
            svo_count = svo_count.join(marginal2[mode_pair], on=mode_pair,
                                       rsuffix='_{}'.format(mode_pair))
        logging.info('Computing association scores..')
        log_total = np.log2(svo_count.freq.sum())
        i_marginal_start = 1 if len(self.part) <= 2 else 4
        for name in svo_count.columns[i_marginal_start:]:
            # Computing log-probabilities  or 1- and 2-marginals
            # TODO cutoff == 0 -> log(0)
            svo_count['log_prob_{}'.format(name)] = np.log2(svo_count[name])
            svo_count['log_prob_{}'.format(name)] -= log_total
        svo_count['log_freq'] = np.log2(svo_count.freq)
        svo_count['log_prob'] = svo_count.log_freq - log_total
        svo_count['pmi'] = svo_count.log_prob
        svo_count['iact_info'] = -svo_count.log_prob
        for mode in self.modes:
            svo_count.pmi -= svo_count['log_prob_freq_{}'.format(mode)]
            svo_count.iact_info += svo_count['log_prob_freq_{}'.format(mode)]
        for mode_pair in itertools.combinations(self.modes, 2):
            svo_count.iact_info -= svo_count['log_prob_freq_{}'.format(mode_pair)]
        svo_count['0'] = 0
        svo_count.pmi = svo_count[['pmi', '0']].max(axis=1)
        svo_count.iact_info = svo_count[['iact_info', '0']].max(axis=1)
        del svo_count['0']
        # TODO Interpretation of positive pointwise interaction information
        #logging.debug('Computing salience..')
        svo_count['salience'] = svo_count.pmi * svo_count.log_freq
        svo_count['iact_sali'] = svo_count.iact_info * svo_count.log_freq
        #logging.debug('Computing Dice..')
        svo_count['log_dice'] = 3 * svo_count.freq
        # This is only the numerator of Dice, and no logarithm at this point.
        svo_count['dice_denom'] = 0
        for mode in self.modes:
            svo_count.dice_denom += svo_count['freq_{}'.format(mode)]
        svo_count.log_dice /= svo_count.dice_denom
        del svo_count['dice_denom']
        svo_count.log_dice = np.log2(svo_count.log_dice) 
        svo_count.log_dice += svo_count.log_dice.min()
        logging.info('Saving to {}{}..'.format(self.assoc_df_filen_patt,
                                               self.part))
        svo_count.to_pickle(self.assoc_df_filen_patt.format(self.part, 'pkl'))
        if write_tsv:
            svo_count.to_csv(self.assoc_df_filen_patt.format(self.part, 'tsv'),
                             sep='\t', index=False, float_format='%.5g')
        return svo_count

    def get_sparse(self, weight, cutoff):
        if os.path.exists(self.sparse_filen):
            logging.info('Loading tensor..')
            self.sparse_tensor, self.index =  pickle.load(open(
                self.sparse_filen, mode='rb'))
            return
        if os.path.exists(self.assoc_df_filen_patt.format(self.part, 'pkl')):
            logging.info('Reading association weights from {} {}..'.format(
                self.assoc_df_filen_patt, self.part))
            self.pmi_df = pd.read_pickle(
                self.assoc_df_filen_patt.format(self.part, 'pkl'))
        else:
            self.pmi_df = self.append_pmi()
        self.pmi_df = self.pmi_df.reset_index()
        df = self.pmi_df[self.pmi_df.freq >= cutoff].copy()
        logging.info('Preparing the index.. (weight={})'.format(weight))
        self.index = {}
        for mode in self.modes:
            marginal = -df.groupby(mode)['freq'].sum()
            self.index[mode] = bidict((w, i) for i, w in enumerate(
                [np.nan] +
                list(marginal[marginal.argsort()].index)))
            df['{}_i'.format(mode)] = df[mode].apply(self.index[mode].get)
        logging.debug('Creating tensor (1/3)..')
        coords = df[['{}_i'.format(mode)
                     for mode in self.modes]].T.to_records(index=False)
        logging.debug('Creating tensor (2/3)..')
        coords = tuple(map(list, coords))
        data = df[weight].values
        shape=tuple(len(self.index[mode]) for mode in self.modes)
        logging.debug('Creating tensor (3/3) {}..'.format(shape))
        self.sparse_tensor = sktensor.sptensor(coords, data, shape=shape)
        pickle.dump((self.sparse_tensor, self.index), open(os.path.join(
            self.tensor_dir, self.sparse_filen), mode='wb'))

    def decomp(self, weight, cutoff, rank):
        if cutoff == 0:
            logging.warning('Not implemented, log(0)=?')
        logging.info((weight, rank, cutoff))
        decomp_filen = os.path.join(
            self.tensor_dir,
            '{}_{}_{}_{}.pkl').format('ktensor', weight, cutoff,
                                      rank)
        if os.path.exists(decomp_filen):
            logging.warning('File exists')
            return
        self.sparse_filen = os.path.join(
            self.tensor_dir,
            '{}_{}_{}.pkl'.format( 'sparstensr', weight, cutoff))
        self.get_sparse(weight, cutoff)
        logging.debug('Orth-ALS..')
        result = orth_als(self.sparse_tensor, rank)
        pickle.dump(result, open(decomp_filen, mode='wb'))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Decompose a tensor of verb and argument cooccurrences')
    parser.add_argument(
        '--weight', default='log_freq',
        help="['log_freq', 'pmi', 'iact_info', 'salience', 'iact_sali', "
        "'log_dice']")
    parser.add_argument('--cutoff', type=int, default=2)
    parser.add_argument('--rank', type=int, default=64)
    parser.add_argument('--input-part', default='', dest='input_part')
    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(levelname)-8s [%(lineno)d] %(message)s')
    args = parse_args()
    decomposer = VerbTensor(args.input_part)
    for rank_exp in range(1, 9):
        args.rank = 2**rank_exp
        #decomposer.decomp(weight=args.weight, cutoff=args.cutoff, rank=args.rank)
        #for weight in [ 'log_freq', 'pmi', 'iact_info', 'salience',
        #'iact_sali', 'log_dice']:
        decomposer.decomp(weight='iact_sali', cutoff=1, rank=2**rank_exp)
