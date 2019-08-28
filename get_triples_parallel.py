#!/usr/bin/env python
# coding: utf-8


from functools import reduce
import os
import sys

import pandas as pd

import logging
logging.basicConfig(level=logging.INFO, 
                    format='%(levelname)-8s [%(lineno)d] %(message)s')


df_dict = '/mnt/permanent/home/makrai/project/verb-tensor/top_level/dataframe/'

def sum_frames(trailing_zeros='00'):
    logging.info('Reading dataframes..')
    dfs = [pd.read_pickle(os.path.join(df_dict, 'freq{}{}.pkl').format(i, trailing_zeros)).set_index(
        ['nsubj', 'ROOT', 'dobj']) 
           for i in range(10)]
    logging.info('Summing dataframes..')
    df = reduce(lambda x, y: x.add(y, fill_value=0), dfs)
    logging.info('Pickling dataframes..')
    df = df.astype(int)
    df = df.sort_values('freq', ascending=False)
    df.to_pickle(os.path.join(df_dict, 'freq{}.pkl').format(trailing_zeros))


if __name__ == '__main__':
    if len(sys.argv) > 1: 
        sum_frames(trailing_zeros=sys.argv[1])
    else:
        sum_frames() 
