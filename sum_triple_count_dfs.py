#!/usr/bin/env python
# coding: utf-8


from functools import reduce
import os
import sys

import pandas as pd

import logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(levelname)-8s [%(lineno)d] %(message)s')


df_dir = '/mnt/permanent/home/makrai/project/verb-tensor/optional_dep/dataframe/'

def sum_frames(common_suff):
    filen = os.path.join(df_dir, 'freq0{}.pkl').format(common_suff)
    logging.info(filen)
    df = pd.read_pickle(filen).reset_index().set_index(['nsubj', 'ROOT', 'dobj'])
    for i in range(1, 10):
        filen = os.path.join(df_dir, 'freq{}{}.pkl').format(i, common_suff)
        if not os.path.isfile(filen):
            logging.warning('File {} not exists.'.format(filen))
            continue
        logging.info(filen)
        df0 = pd.read_pickle(filen).reset_index().set_index(['nsubj', 'ROOT', 'dobj'])
        df += df0.reindex(df.index, fill_value=0)
    logging.info('Pickling dataframes..')
    df = df.astype(int)
    df = df.sort_values('freq', ascending=False)
    df.to_pickle(os.path.join(df_dir, 'freq{}.pkl').format(common_suff))


if __name__ == '__main__':
    sum_frames(common_suff=sys.argv[1])
