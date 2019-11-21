#!/usr/bin/env python
# coding: utf-8

from collections import defaultdict
import numpy as np
import operator
import os
import pandas as pd
import pickle

import logging
logging.basicConfig(level=logging.DEBUG,
        format='%(levelname)-8s [%(lineno)d] %(message)s')


tensor_dir = '/mnt/permanent/home/makrai/project/verb-tensor/verb/tensor/0/'
verb_sim_data_dir = '/mnt/permanent/Language/English/Data/verb-similarity/Sadrzadeh/'
test_data_dir = '/mnt/permanent/Language/English/Data'
consistent_name_d = {
    'pmi': 'pmi-vanl', 'iact_info': 'iact-vanl', 'log_dice': 'dice-vanl',
    'salience': 'pmi-sali', 'iact_sali': 'iact-sali', 'dice_sali': 'dice-sali'}


def test_sim(task_df, cutoff=5, max_rank=128, mode_to_test='svo',
             normlz_vocb=True, lmbda=False): 
    modes = ['nsubj', 'ROOT', 'dobj']
    if mode_to_test == 'svo':
        logging.info('Assuming df is Kartsaklis and Sadrzadeh Turk')
        query1_cols = list(enumerate(['subject1', 'verb1', 'object1']))
        query2_cols = list(enumerate(['subject2', 'verb2', 'object2']))
        sim_col = 'score' 
    elif mode_to_test == 'ROOT':
        logging.info('Assuming df is SimVerb')
        query1_cols = [(1, 'verb1')]
        query2_cols = [(1, 'verb2')]
        sim_col = 'sim' 
    elif mode_to_test == 'nsubj': 
        logging.info('Assuming df is SimLex')
        query1_cols = [(0, 'word1')]
        query2_cols = [(0, 'word2')]
        sim_col = 'SimLex999' 
    elif mode_to_test == 'dobj': 
        logging.info('Assuming df is SimLex')
        query1_cols = [(2, 'word1')]
        query2_cols = [(2, 'word2')]
        sim_col = 'SimLex999' 
    else:
        raise Exception(
            'mode_to_test has to be eigther svo, nsubj, ROOT, or dobj')
    mean = task_df[sim_col].mean()
    for weight_oldname, weight_newname in consistent_name_d.items():
        logging.info('Testing {}..'.format(weight_newname))
        basen = 'sparstensr_{}_{}.pkl'.format(weight_oldname, cutoff)
        stensor, index = pickle.load(open(os.path.join(tensor_dir, basen),
                                          mode='rb'))
        for exp in range(1, int(np.log2(max_rank))+1):
            rank = 2**exp
            oov = defaultdict(int)
            target_col = 'tensor_sim_{}_{}'.format(weight_newname, rank)
            try:
                basen = 'ktensor_{}_{}_{}.pkl'.format(weight_oldname, cutoff, rank)
                ktensor, fit, n_iterations, exectimes = pickle.load(open(
                    os.path.join(tensor_dir, basen), mode='rb'))
            except FileNotFoundError as e:
                logging.warning(e)
                task_df[target_col] = mean
                continue
            if lmbda:
                sq_lam = np.sqrt(np.apply_along_axis(np.linalg.norm, 0, ktensor.lmbda))
                for mode_i in modes_used:
                    ktensor.U[mode_i] *= sq_lam
            if normlz_vocb and mode_to_test != 'svo':
                mode_i = query1_cols[0][0]
                ktensor.U[mode_i] /= np.apply_along_axis(
                    np.linalg.norm, 1, ktensor.U[mode_i]).reshape((-1,1))
            def lookup(word, mode_i=1):
                try:
                    return ktensor.U[mode_i][index[modes[mode_i]][word]]
                except KeyError as e:
                    oov[e.args] += 1
                    return np.zeros(rank)
            for mode_i, query_w_col in query1_cols + query2_cols:
                series = task_df[query_w_col].apply( 
                    lambda word: lookup(word, mode_i=mode_i))
                task_df['{}_v'.format(query_w_col)] = series
            if mode_to_test == 'svo':
                for qwocs, svo_vc in [(query1_cols, 'svo1_v'),
                                                (query2_cols, 'svo2_v')]:
                    qvecs = ['{}_v'.format(qwc) for qwc in list(zip(*qwocs))[1]]
                    task_df[svo_vc] = task_df[qvecs].apply(np.concatenate, axis=1)
            if normlz_vocb and mode_to_test == 'svo':
                for svo_vc in ['svo1_v', 'svo2_v']:
                    task_df[svo_vc] /= task_df[svo_vc].apply(np.linalg.norm)
            if mode_to_test == 'svo':
                query_v_cols = ['svo1_v', 'svo2_v'] 
            else:
                query_v_cols = [
                    '{}_v'.format(item[0][1]) 
                                  for item in [query1_cols, query2_cols]]
            def cell_dot_cell(series):
                return series[0].dot(series[1])
            task_df[target_col] = task_df[query_v_cols].apply(cell_dot_cell, axis=1)
            if rank == max_rank:
                logging.debug(sorted(oov.items(), key=operator.itemgetter(1),
                                     reverse=True)[:5])
            logging.debug(task_df.corr(method='spearman').loc[sim_col].sort_values())#ascending=False))
    return task_df.corr(method='spearman').loc[sim_col].sort_values(
        ascending=False)


def read_sim_data(filen):
    return pd.read_csv(os.path.join(verb_sim_data_dir, filen), sep=' ')


def read_ks():
    cols = ['subject1', 'verb1', 'object1', 'subject2', 'verb2', 'object2']
    return read_sim_data('emnlp2013_turk.txt').groupby(cols).mean().drop(
        columns=['annotator'])

def read_SimVerb():
    return pd.read_csv( 
        os.path.join(test_data_dir,
                     '/verb-similarity/simverb-3500/SimVerb-3500.txt'),
        sep='\t', header=None, names=['verb1', 'verb2', 'pos', 'sim', 'rel'])

def read_SimLex():
    return pd.read_csv(os.path.join(test_data_dir,
                                    'SimLex-999/SimLex-999.txt'), sep='\t')


if __name__ == '__main__':
    test_sim(read_ks().reset_index(), mode_to_test='svo')
    #test_sim(read_SimVerb().reset_index(), mode_to_test='ROOT')
    #test_sim(read_SimLex().reset_index(), mode_to_test='nsubj')
    #test_sim(read_SimLex().reset_index(), mode_to_test='dobj')


def predict_verb(target_df, weight, rank, cutoff=5, prec_at=1, log_oov=False):
    _, index = pickle.load(open(os.path.join(
        tensor_dir, 'sparstensr_{}_{}.pkl').format(weight, cutoff), mode='rb'))
    basen = 'ktensor_{}_{}_{}.pkl'.format(weight, cutoff, rank)
    ktensor, fit, n_iterations, exectimes = pickle.load(open(os.path.join(
        tensor_dir, basen), mode='rb'))
    def lookup(word, mode_i=1, deprel='ROOT'):
        """
        modes are ['nsubj', 'ROOT', 'dobj'].
        """
        return ktensor.U[mode_i][index[deprel][word]]
    logging.debug('Making predictions..')
    # TODO ktensor.U[mode] *= sq_lam
    oov = defaultdict(int)
    def verb_pred(series):
        i = '' # or '1'
        series = series[['subject{}'.format(i), 'object{}'.format(i)]]
        try:
            predicted_ids = np.argsort(
                (-ktensor.lmbda * 
                 lookup(series[0], mode_i=0, deprel='nsubj')) .dot( 
                    (ktensor.U[1] * 
                     lookup(series[1], mode_i=2, deprel='dobj')).T))
            return [index['ROOT'].inverse[i] for i in predicted_ids[:prec_at]]
        except KeyError as e:
            oov[e.args] += 1
            return []
    target_df['predicted_{}_{}'.format(weight, rank)] = target_df.apply(
        verb_pred, axis=1)
    if log_oov:
        logging.debug(sorted(oov.items(), key=operator.itemgetter(1),
                             reverse=True))
    logging.debug('Evaluating predictions..')
    for target in ['landmark', 'verb']:
        def is_good(series):
            return series[target] in series['predicted_{}_{}'.format(weight, rank)]
        target_df['good_{}_{}_{}'.format(target, weight, rank)] = target_df.apply(is_good, axis=1)
        n_good = target_df['good_{}_{}_{}'.format(target, weight, rank)].sum()
        major_baseline = 130 if target == 'landmark' else 260
        if n_good > major_baseline:
            logging.info('{}\t{}\t{}\t{}'.format(target, weight, rank, n_good))
            

def df_columns_from_filen(sim_df, verb=True):
    if verb:
        sim_df = sim_df.drop('sim')
    sim_df = sim_df.to_frame().reset_index()
    sim_df [['weight', 'rank_']] = pd.DataFrame(sim_df['index'].str.rsplit('_', 1).values.tolist())
    sim_df.rank_ = sim_df.rank_.astype(int)
    sim_df['weight'] = pd.DataFrame(sim_df['weight'].str.split('_', 2).values.tolist())[2]
    sim_df = sim_df.drop(columns='index')
    sim_df = sim_df.drop(labels=[0])
    sim_df =sim_df[sim_df.isna().sum(axis=1)==0]
    return sim_df.sort_values('sim' if verb else 'SimLex999', ascending=False)
