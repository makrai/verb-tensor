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
#from tensorly.contrib.sparse.decomposition import parafac
from tensorly import tensor
from tensorly.decomposition import parafac


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
        filen = os.path.join(self.project_dir, f'dataframe/freq{self.input_part}.pkl')
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
        df['iact_info'] = -df.log_prob
        for mode in self.modes:
            df.pmi -= df[f'log_prob_freq_{mode}']
            df.iact_info -= df[f'log_prob_freq_{mode}']
        for mode_pair in itertools.combinations(self.modes, 2):
            df.iact_info += df[f'log_prob_freq_{mode_pair}']
        if positive:    
            df['0'] = 0
            df.pmi = df[['pmi', '0']].max(axis=1)
            df.iact_info = df[['iact_info', '0']].max(axis=1)
            del df['0']
        df['npmi'] = df.pmi / -df.log_prob
        df['niact'] = df.iact_info / -df.log_prob
        # TODO Interpretation of positive pointwise interaction information
        #logging.debug('Computing salience..')
        df['salience'] = df.pmi * df.log_freq
        df['iact_sali'] = df.iact_info * df.log_freq
        
        #logging.debug('Computing Dice..')
        df['log_dice'] = 3 * df.freq
        df['dice_denom'] = 0
        for mode in self.modes:
            df.dice_denom += df[f'freq_{mode}']
        df.log_dice /= df.dice_denom
        del df['dice_denom']
        df.log_dice = np.log2(df.log_dice)
        df.log_dice -= df.log_dice.min()

        df['dice_sali'] = df.log_dice * df.log_freq
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
        df = self.pmi_df[self.pmi_df.freq >= cutoff].copy()
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

    def decomp(self, weight, cutoff, rank):
        if cutoff == 0:
            logging.warning('Not implemented, log(0)=?')
        logging.info((weight, rank, cutoff))
        decomp_filen = os.path.join(self.tensor_dir,
                                    f'ktensor_{weight}_{cutoff}_{rank}.pkl')
        if os.path.exists(decomp_filen):
            logging.warning('File exists')
            return
        self.get_sparse(weight, cutoff)
        tl.set_backend('pytorch')
        logging.info(f'Decomposition.. {tl.get_backend()}')
        sparse_tensor = tensor(self.sparse_tensor)
        # device='cuda:0'. Dense is faster on CPU.
        result = parafac(sparse_tensor, rank=rank)
        pickle.dump(result, open(decomp_filen, mode='wb'))


weights =  ['freq', 'log_freq', 'pmi', 'iact_info', 'salience', 'iact_sali',
            'log_dice', 'dice_sali', 'npmi', 'niact']

def parse_args():
    parser = argparse.ArgumentParser(
        description='Decompose a tensor of verb and argument cooccurrences')
    parser.add_argument('--weight', choices=['for', 'rand']+weights)
        #default='log_freq',
    parser.add_argument('--cutoff', type=int, default=5)
    parser.add_argument('--rank', type=int)#, default=64)
    parser.add_argument('--input-part', default='', dest='input_part')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)-8s [%(lineno)d] %(message)s')
    args = parse_args()
    decomposer = VerbTensor(args.input_part)
    if args.weight == 'for':
        logging.info('')
        #for exp in range(1, 10):
        #args.rank = 2**exp#np.random.randint(1, 9)
        for weight in weights: 
            args.weight = weight#s[np.random.randint(0, len(weights))]
            decomposer.decomp(weight=args.weight, cutoff=args.cutoff,
                              rank=args.rank)
    elif args.weight == 'rand':
        #while True:
        #args.rank = 2**np.random.randint(1, 9)
        args.weight = weights[np.random.randint(0, len(weights))]
    decomposer.decomp(weight=args.weight, cutoff=args.cutoff, rank=args.rank)
