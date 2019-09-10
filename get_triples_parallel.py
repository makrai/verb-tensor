#!/usr/bin/env python
# coding: utf-8


from functools import reduce
import os
import sys

import pandas as pd

import logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(levelname)-8s [%(lineno)d] %(message)s')


df_dir = '/mnt/permanent/home/makrai/project/verb-tensor/top_level/dataframe/'

def sum_frames(trailing_zeros='00'):
    filen = os.path.join(df_dir, 'freq0{}.pkl').format(trailing_zeros)
    logging.info(filen)
    df = pd.read_pickle(filen).reset_index().set_index(['nsubj', 'ROOT', 'dobj'])#
    logging.debug(df.head(1))
    for i in range(1, 10):
        filen = os.path.join(df_dir, 'freq{}{}.pkl').format(i, trailing_zeros)
        logging.info(filen)
        df0 = pd.read_pickle(filen).reset_index().set_index(['nsubj', 'ROOT', 'dobj'])
        df += df0.reindex(df.index, fill_value=0)
    logging.info('Pickling dataframes..')
    df = df.astype(int)
    df = df.sort_values('freq', ascending=False)
    df.to_pickle(os.path.join(df_dir, 'freq{}.pkl').format(trailing_zeros))


if __name__ == '__main__':
    if len(sys.argv) > 1: 
        sum_frames(trailing_zeros=sys.argv[1])
    else:
        sum_frames() 
