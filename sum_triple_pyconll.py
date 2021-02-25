#!/usr/bin/env python
# coding: utf-8

import configparser
from functools import reduce
import os
import sys

import pandas as pd

import logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(levelname)-8s [%(lineno)d] %(message)s')

config = configparser.ConfigParser()
config.read('config.ini')
df_dir = config['DEFAULT']['ProjectDirectory']+'dataframe/'


def read_series_or_df(filen):
    data = pd.read_pickle(filen)
    if isinstance(data, pd.Series):
        return data.to_frame(name='freq')
    elif isinstance(data, pd.DataFrame):
        return data.reset_index().set_index(['nsubj', 'ROOT', 'dobj'])
    else:
        raise NotImplementedError


def sum_hunderd_files(i=0):
    """
    13 min
    """
    df = None
    for j in range(10):
        logging.info(f'Adding freq{j}{i}x')
        for k in range(10):
            filen = os.path.join(df_dir, f'freq{k}{j}{i}.pkl')
            if not os.path.isfile(filen):
                logging.warning('File {} not exists.'.format(filen))
                continue
            df0 = read_series_or_df(filen)
            if df is None:
                df = df0
            else:
                df += df0.reindex(df.index, fill_value=0)
    logging.info('Pickling dataframe..')
    df = df.astype(int)
    df = df.sort_values('freq', ascending=False)
    df.to_pickle(os.path.join(df_dir, f'freq{i}.pkl'))
    return df


def sum_second_half():
    df = None
    for i in range(5, 10):
        basen = f'freq{i}.pkl'
        logging.info(f'Adding {basen}..')
        filen = os.path.join(df_dir, basen)
        if not os.path.isfile(filen):
            logging.warning('File {} not exists.'.format(filen))
            continue
        df0 = read_series_or_df(filen)
        if df is None:
            df = df0
        else:
            df += df0.reindex(df.index, fill_value=0)
    logging.info('Pickling dataframe..')
    df = df.astype(int)
    df = df.sort_values('freq', ascending=False)
    df.to_pickle(os.path.join(df_dir, f'freq5to9.pkl'))
    return df


def sum_ten_files(common_suff):
    filen = os.path.join(df_dir, 'freq0{}.pkl').format(common_suff)
    df = read_series_or_df(filen)
    for i in range(1, 10):
        basen = 'freq{}{}.pkl'
        logging.info(basen)
        filen = os.path.join(df_dir, basen).format(i, common_suff)
        if not os.path.isfile(filen):
            logging.warning('File {} not exists.'.format(filen))
            continue
        df0 = read_series_or_df(filen)
        df += df0.reindex(df.index, fill_value=0)
    logging.info('Pickling dataframe..')
    df = df.astype(int)
    df = df.sort_values('freq', ascending=False)
    df.to_pickle(os.path.join(df_dir, 'freq{}.pkl').format(common_suff))
    return df


if __name__ == '__main__':
    #sum_hunderd_files(i=sys.argv[1])
    #sum_second_half()
    sum_ten_files('')
