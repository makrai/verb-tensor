#!/usr/bin/env python
# coding: utf-8


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


def append_pmi(svo_count, modes=['NOM', 'stem', 'ACC'], compute_freq=True,
        debug_index=None):
    if compute_freq:
        logging.info('Computing freq..')
        svo_count = svo_count.groupby(modes).size().reset_index(name='freq')
        svo_count.sort_values('freq', ascending=False).to_csv(
            '/mnt/permanent/Language/Hungarian/Dic/sass15-535k-igei-szerkezet/mazsola_adatbazis_svo_freq.tsv',
            sep='\t', index=False)
    logging.info('Computing marginals..')
    marginal = {mode: svo_count.groupby(mode).sum() for mode in modes}
    logging.info('Computing 2-marginals..')
    marginal2 = {mode_pair: svo_count.groupby(list(mode_pair)).sum()
             for mode_pair in itertools.combinations(modes, 2)}
    log_total = np.log2(svo_count.freq.sum())
    for mode in modes:
        svo_count = svo_count.join(marginal[mode], on=mode,
                rsuffix='_{}'.format(mode))
    logging.info('Computing Dice..')
    for mode_pair in itertools.combinations(modes, 2):
        svo_count = svo_count.join(marginal2[mode_pair], on=mode_pair,
                rsuffix='_{}'.format(mode_pair))
    svo_count['dice'] = 3*svo_count.freq # This is only the numerator.
    svo_count['freq2'] = svo_count.freq
    if debug_index:
        logging.debug(svo_count.loc[debug_index])
    svo_count['dice_denom'] = 0
    for mode in modes:
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
    for mode in modes:
        svo_count.pmi -= svo_count['freq_{}'.format(mode)]
        svo_count.iact_info += svo_count['freq_{}'.format(mode)]
        if debug_index:
            logging.debug(svo_count.loc[debug_index])
    svo_count.pmi = np.max(svo_count.pmi, 0)
    # TODO positive iact, sali, logDice..
    for mode_pair in itertools.combinations(modes, 2):
        svo_count.iact_info -= svo_count['freq_{}'.format(mode_pair)]
        if debug_index:
            logging.debug(svo_count.loc[debug_index])
    svo_count['salience'] = svo_count.pmi * svo_count.freq2
    svo_count['iact_sali'] = svo_count.iact_info * svo_count.freq2
    return svo_count, log_total


def get_tensor(df, column):
    logging.debug(column)
    modes=['NOM', 'stem', 'ACC']
    index = {mode: bidict(df.groupby(mode)[column].sum().argsort().to_dict())
            for mode in modes}
    for mode in modes:
        index[mode][np.nan] = len(index[mode])
        df['{}_i'.format(mode)] = df[mode].apply(index[mode].get)
    logging.debug('')
    x = df[['{}_i'.format(mode) for mode in modes]].T.to_records(index=False)
    coords = tuple(map(list, x))
    data = df[column].values
    shape=tuple(len(index[mode]) for mode in modes)
    logging.debug(([len(y) for y in coords], len(data)))
    logging.info(shape)
    return sktensor.sptensor(coords, data, shape=shape), index


def decomp(mazsola_df, column='freq2', cutoff=1, rank=100):
    logging.debug((cutoff, rank))
    pmi_dir = '/mnt/store/home/makrai/project/verb-tensor/pmi/'
    vtensor, index = get_tensor(mazsola_df[mazsola_df.freq>cutoff].copy(), 
            column=column)
    pickle.dump((vtensor, index), open(os.path.join(pmi_dir,
        '{}_{}_{}_{}.pkl').format('sparstensr', column, cutoff, rank), mode='wb'))
    result = orth_als(vtensor, rank)
    pickle.dump(result, open(os.path.join(pmi_dir,
        '{}_{}_{}_{}.pkl').format('ktensor', column, cutoff, rank), mode='wb'))


if __name__ == '__main__':
    mazsola_df = pd.read_csv(
            '/mnt/permanent/Language/Hungarian/Dic/sass15-535k-igei-szerkezet/mazsola_adatbazis_svo_freq.tsv',
            sep='\t')
    mazsola_df, log_total = append_pmi(mazsola_df, compute_freq=False)
    for cutoff_exp in range(6, -1, -1):
        decomp(mazsola_df, cutoff=2**cutoff_exp)
