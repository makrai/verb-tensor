#!/usr/bin/env python
# coding: utf-8


import argparse
import itertools
import os
import pickle
import logging

from bidict import bidict
import configparser
import numpy as np
import pandas as pd
import sparse
import tensorly as tl

# Use two of the following four
#from tensorly.contrib.sparse import tensor
from tensorly.contrib.sparse.decomposition import tucker, non_negative_tucker
from tensorly import tensor
from tensorly.contrib.sparse.decomposition import parafac


class VerbTensor():
    def __init__(self, input_part):
        self.input_part = input_part
        config = configparser.ConfigParser()
        config.read('config.ini')
        self.project_dir = config['DEFAULT']['ProjectDirectory']
        self.tensor_dir = os.path.join(self.project_dir, 'tensor',
                                       self.input_part)
        self.assoc_df_filen_patt = os.path.join(self.project_dir,
                                                'dataframe/assoc{}.{}')
        self.modes = ['nsubj', 'ROOT', 'dobj']
        # mazsola: ['NOM', 'stem', 'ACC']

    def append_pmi(self, write_tsv=False, positive=True):
        filen = os.path.join(self.project_dir,
                             f'dataframe/freq{self.input_part}.pkl')
        logging.info(f'Reading freqs from {filen}')
        df = pd.read_pickle(filen)
        logging.info('Computing marginals..')
        marginal = {mode: df.groupby(mode).sum() for mode in self.modes}
        marginal2 = {mode_pair: df.groupby(list(mode_pair)).sum()
                     for mode_pair in itertools.combinations(self.modes, 2)}
        for mode in self.modes:
            df = df.join(marginal[mode], on=mode, rsuffix=f'_{mode}')
        for mode_pair in itertools.combinations(self.modes, 2):
            logging.info(mode_pair)
            df = df.join(marginal2[mode_pair], on=mode_pair,
                         rsuffix=f'_{mode_pair}')
        logging.info('Computing association scores..')
        log_total = np.log2(df.freq.sum())
        i_marginal_start = 1 #if len(self.input_part) <= 2 else 4
        for name in df.columns[i_marginal_start:]:
            # Computing log-probabilities or 1- and 2-marginals
            # TODO cutoff == 0 -> log(0)
            df[f'log_prob_{name}'] = np.log2(df[name])
            df[f'log_prob_{name}'] -= log_total
        df['log_freq'] = np.log2(df.freq)
        df['log_prob'] = df.log_freq - log_total
        df['pmi'] = df.log_prob
        df['iact'] = -df.log_prob
        for mode in self.modes:
            df.pmi -= df[f'log_prob_freq_{mode}']
            df.iact -= df[f'log_prob_freq_{mode}']
        for mode_pair in itertools.combinations(self.modes, 2):
            df.iact += df[f'log_prob_freq_{mode_pair}']
        if positive:
            df['0'] = 0
            df.pmi = df[['pmi', '0']].max(axis=1)
            df.iact = df[['iact', '0']].max(axis=1)
            del df['0']
        df['npmi'] = df.pmi / -df.log_prob
        df['niact'] = df.iact / -df.log_prob
        # TODO Interpretation of positive pointwise interaction information
        #logging.debug('Computing pmi_sali..')
        df['pmi_sali'] = df.pmi * df.log_freq
        df['iact_sali'] = df.iact * df.log_freq

        #logging.debug('Computing Dice..')
        df['ldice'] = df.freq
        df['dice_denom'] = 0
        for mode in self.modes:
            df.dice_denom += df[f'freq_{mode}']
        df.ldice /= df.dice_denom
        del df['dice_denom']
        df.ldice = np.log2(df.ldice)
        df.ldice -= df.ldice.min()

        df['ldice_sali'] = df.ldice * df.log_freq
        logging.info(f'Saving to {self.assoc_df_filen_patt}{self.input_part}..')
        df.to_pickle(self.assoc_df_filen_patt.format(self.input_part, 'pkl'))
        if write_tsv:
            df.to_csv(self.assoc_df_filen_patt.format(self.input_part, 'tsv'),
                      sep='\t', index=False, float_format='%.5g')
        return df

    def get_sparse(self, weight, cutoff):
        self.sparse_filen = os.path.join(
            self.tensor_dir,
            f'sparstensr_{weight}_{cutoff}.pkl')
        if os.path.exists(self.sparse_filen):
            logging.info('Loading tensor..')
            self.sparse_tensor, self.index =  pickle.load(open(
                self.sparse_filen, mode='rb'))
            return
        if os.path.exists(self.assoc_df_filen_patt.format(self.input_part,
                                                          'pkl')):
            logging.info('Reading association weights from '+
                         self.assoc_df_filen_patt.format(self.input_part, '.'))
            self.pmi_df = pd.read_pickle(
                self.assoc_df_filen_patt.format(self.input_part, 'pkl'))
        else:
            self.pmi_df = self.append_pmi()
        self.pmi_df = self.pmi_df.reset_index()
        freq_sub_tensor = self.pmi_df[ 
                (self.pmi_df.freq_nsubj >= cutoff) &
                (self.pmi_df.freq_ROOT >= cutoff) & 
                (self.pmi_df.freq_dobj >= cutoff)] 
        df = freq_sub_tensor.copy()
        logging.info(f'Preparing the index.. (weight={weight})')
        self.index = {}
        for mode in self.modes:
            marginal = -df.groupby(mode)['freq'].sum()
            self.index[mode] = bidict((w, i) for i, w in enumerate(
                #[np.nan] +
                list(marginal[marginal.argsort()].index)))
            df[f'{mode}_ind'] = df[mode].apply(self.index[mode].get)
        logging.info('Creating tensor (1/3)..')
        coords = df[[f'{mode}_ind'
                     for mode in self.modes]].T.to_records(index=False)
        logging.info('Creating tensor (2/3)..')
        coords = tuple(map(list, coords))
        data = df[weight].values
        shape=tuple(len(self.index[mode]) for mode in self.modes)
        logging.info(
            'Creating tensor with shape {} and {} nonzeros..  (3/3)'.format(
                ' x '.join(map(str, shape)),
                len(np.nonzero(data)[0])))
        self.sparse_tensor = sparse.COO(coords, data, shape=shape)
        pickle.dump((self.sparse_tensor, self.index),
                    open(os.path.join(self.tensor_dir, self.sparse_filen),
                         mode='wb'))

    def decomp(self, weight, cutoff, rank, decomp_algo, non_negative):
        if cutoff == 0:
            logging.warning('Not implemented, log(0)=?')
        logging.info((weight, rank, cutoff))
        non_neg_str = 'non_negative_' if non_negative else '' 
        decomp_filen = os.path.join(self.tensor_dir,
                                    f'{non_neg_str}{decomp_algo}_{weight}_{cutoff}_{rank}.pkl')
        if os.path.exists(decomp_filen):
            logging.warning('File exists')
            return
        self.get_sparse(weight, cutoff)
        logging.info(self.sparse_tensor.shape)
        logging.info(f'Decomposition..')
        if decomp_algo == 'tucker':
            rank = map(int, rank.split(',')) if ',' in rank else int(rank)
            if non_negative:
                result = non_negative_tucker(self.sparse_tensor, rank=rank)
            else:
                result = tucker(self.sparse_tensor, rank=rank)
        else:
            rank = int(rank)
            if non_negative:
                raise NotImplementedError
            #tl.set_backend('pytorch')
            result = parafac(self.sparse_tensor, init='random', rank=rank,
                             verbose=True)
            # tensor(.., device='cuda:0')
        pickle.dump(result, open(decomp_filen, mode='wb'))


weights =  ['log_freq', 'pmi', 'iact', 'pmi_sali', 'iact_sali',
            'ldice', 'ldice_sali', 'npmi', 'niact'] # freq TODO

def parse_args():
    parser = argparse.ArgumentParser(
        description='Decompose a tensor of verb and argument cooccurrences')
    parser.add_argument('--non_negative', action='store_true')
    parser.add_argument('--decomp_algo', choices=['tucker', 'parafac'],
        default='tucker')
    parser.add_argument('--rank', default=64)
    parser.add_argument('--cutoff', type=int, default=100000)
    parser.add_argument('--weight', choices=['for', 'rand']+weights,
            default='log_freq')
    parser.add_argument('--input-part', default='', dest='input_part')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)-8s [%(lineno)d] %(message)s')
    args = parse_args()
    decomposer = VerbTensor(args.input_part)
    if args.weight == 'for':
        #for exp in range(1, 10):
        #args.rank = 2**exp#np.random.randint(1, 9)
        for weight in weights:
            args.weight = weight#s[np.random.randint(0, len(weights))]
            decomposer.decomp(weight=args.weight, cutoff=args.cutoff,
                              rank=args.rank, decomp_algo=args.decomp_algo,
                              non_negative=args.non_negative)
    elif args.weight == 'rand':
        #while True:
        #args.rank = 2**np.random.randint(1, 9)
        args.weight = weights[np.random.randint(0, len(weights))]
    decomposer.decomp(
            weight=args.weight, cutoff=args.cutoff, rank=args.rank,
            decomp_algo=args.decomp_algo, non_negative=args.non_negative)
