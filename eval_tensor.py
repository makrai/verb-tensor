#!/usr/bin/env python
# coding: utf-8

from collections import defaultdict
import configparser
import logging
import numpy as np
import operator
import os
import pandas as pd
import pickle
import tensorly as tl

import bidict

config = configparser.ConfigParser()
config.read('config.ini')
tensor_dir = config['DEFAULT']['ProjectDirectory']+'tensor/'
test_data_dir = '/mnt/permanent/Language/English/Data/'
verb_sim_data_dir = f'{test_data_dir}verb-similarity/Sadrzadeh/'
consistent_name_d = {
    'freq': 'freq_vanl',
    'pmi': 'pmi_vanl', 'iact_info': 'iact_vanl', 'log_dice': 'dice_vanl', # vanl
    'npmi': 'pmi_norm', 'niact': 'iact_norm', # normed
    'salience': 'pmi_sali', 'iact_sali': 'iact_sali', 'dice_sali': 'dice_sali'}


def read_sim_data(filen):
    return pd.read_csv(os.path.join(verb_sim_data_dir, filen), sep=' ')


def read_ks():
    cols = ['subject1', 'verb1', 'object1', 'subject2', 'verb2', 'object2']
    return read_sim_data('emnlp2013_turk.txt').groupby(cols).mean().drop(
        columns=['annotator'])

def read_gs():
    cols = ['verb', 'subject', 'object', 'landmark', 'hilo']
    return read_sim_data('GS2011data.txt').groupby(cols).mean()


def read_SimVerb():
    return pd.read_csv(
        os.path.join(test_data_dir,
                     '/verb-similarity/simverb-3500/SimVerb-3500.txt'),
        sep='\t', header=None, names=['verb1', 'verb2', 'pos', 'sim', 'rel'])


def read_SimLex():
    return pd.read_csv(os.path.join(test_data_dir,
                                    'SimLex-999/SimLex-999.txt'), sep='\t')


class VerbTensorEvaluator():
    def get_cols(self, mode_to_test):
        if mode_to_test == 'svo':
            #logging.info('Assuming df is Kartsaklis and Sadrzadeh Turk')
            self.query1_cols = list(enumerate(['subject1', 'verb1', 'object1']))
            self.query2_cols = list(enumerate(['subject2', 'verb2', 'object2']))
            self.sim_col = 'score'
        elif mode_to_test == 'ROOT':
            #logging.info('Assuming df is SimVerb')
            self.query1_cols = [(1, 'verb1')]
            self.query2_cols = [(1, 'verb2')]
            self.sim_col = 'sim'
        elif mode_to_test == 'nsubj':
            #logging.info('Assuming df is SimLex')
            self.query1_cols = [(0, 'word1')]
            self.query2_cols = [(0, 'word2')]
            self.sim_col = 'SimLex999'
        elif mode_to_test == 'dobj':
            #logging.info('Assuming df is SimLex')
            self.query1_cols = [(2, 'word1')]
            self.query2_cols = [(2, 'word2')]
            self.sim_col = 'SimLex999'
        else:
            raise Exception(
                'mode_to_test has to be eigther svo, nsubj, ROOT, or dobj')

    def load_embeddings(self, decomp_algo='tucker', weight='log_freq',
            cutoff=2000, rank=128, normlz_vocb_and_not_svo=False, lmbda=False,
            non_negative=False):
        _, self.index = pickle.load(open(os.path.join(
            tensor_dir, f'sparstensr_{weight}_{cutoff}.pkl'), mode='rb'))
        non_negative = 'non_negative_' if non_negative else ''
        basen = f'{non_negative}{decomp_algo}_{weight}_{cutoff}_{rank}.pkl'
        self.decomped_tns = pickle.load(open(os.path.join(tensor_dir, basen), mode='rb'))
        factors = self.decomped_tns.factors
        if decomp_algo == 'parafac':
            factors = [factor.todense() for factor in factors]
        if lmbda:
            sq_lam = np.sqrt(np.apply_along_axis(np.linalg.norm, 0,
                                                 self.decomped_tns.lmbda))
            for mode_i in modes_used:
                factors[mode_i] *= sq_lam
        if normlz_vocb_and_not_svo:
            mode_i = self.query1_cols[0][0]
            factors[mode_i] /= np.apply_along_axis(
                np.linalg.norm, 1, factors[mode_i]).reshape((-1,1))
        modes = ['nsubj', 'ROOT', 'dobj']
        self.oov = defaultdict(int)
        def lookup(word, mode_i=1):
            try:
                return factors[mode_i][self.index[modes[mode_i]][word]]
            except KeyError as e:
                self.oov[e.args] += 1
                return np.zeros(rank)
        self.lookup = lookup

    def test_sim(self, task_df0, cutoff=2000, rank=128, mode_to_test='svo',
                 normlz_vocb=True, lmbda=False, decomp_algo='tucker',
                 weight='log_freq', non_negative=False):
        task_df = task_df0.copy() # Subj and obj sim not to go in the same df
        self.get_cols(mode_to_test)
        mean = task_df[self.sim_col].mean()
        self.load_embeddings(decomp_algo=decomp_algo, weight=weight,
                cutoff=cutoff, rank=rank, 
                normlz_vocb_and_not_svo=normlz_vocb and mode_to_test!='svo',
                lmbda=lmbda, non_negative=non_negative)
        for mode_i, query_w_col in self.query1_cols + self.query2_cols:
            series = task_df[query_w_col].apply(
                lambda word: self.lookup(word, mode_i=mode_i))
            task_df['{}_v'.format(query_w_col)] = series
        if mode_to_test == 'svo':
            for qwocs, svo_vc in [(self.query1_cols, 'svo1_v'),
                                  (self.query2_cols, 'svo2_v')]:
                qvecs = [f'{qwc[1]}_v' for qwc in qwocs]
                task_df[svo_vc] = task_df[qvecs].apply(np.concatenate, axis=1)
        if normlz_vocb and mode_to_test == 'svo':
            for svo_vc in ['svo1_v', 'svo2_v']:
                task_df[svo_vc] /= task_df[svo_vc].apply(np.linalg.norm)
        if mode_to_test == 'svo':
            query_v_cols = ['svo1_v', 'svo2_v']
        else:
            query_v_cols = [f'{item[0][1]}_v'
                            for item in [self.query1_cols, self.query2_cols]]
        def cell_dot_cell(series):
            return series[0].dot(series[1])
        target_col = 'tensor_sim'#_{weight}_{cutoff}_{rank}'
        task_df[target_col] = task_df[query_v_cols].apply(cell_dot_cell, axis=1)
        logging.debug(sorted(self.oov.items(), key=operator.itemgetter(1),
                             reverse=True)[:5])
        sim_corr = task_df.corr(method='spearman').loc[self.sim_col]
        logging.debug(sim_corr)
        return sim_corr[target_col]

    def predict_verb(self, target_df, weight='log_freq', rank=128, cutoff=2000,
            prec_at=1, logg_oov=False):
        """
        Results are bellow the majority baseline.
        """
        self.load_embeddings()
        #logging.debug('Making predictions..')
        low_verb_low = tl.tenalg.mode_dot(
                self.decomped_tns.core, self.decomped_tns.factors[1], mode=1)
        cols_suff = '' # GS11 or Jenatton: '', KS13: 1 or 2
        def verb_pred(series):
            series = series[[f'subject{cols_suff}', f'object{cols_suff}']]
            v_subj = self.lookup(series[0], mode_i=0)
            v_obj = self.lookup(series[1], mode_i=2)
            assocv = tl.tenalg.mode_dot(low_verb_low, v_subj, mode=0).dot(v_obj)
            predicted_ids = np.argsort(-assocv)
            return [self.index['ROOT'].inverse[cols_suff] 
                    for cols_suff in predicted_ids[:prec_at]]
        target_df['predicted'] = target_df.apply(verb_pred, axis=1)
        if logg_oov:
            logging.debug(sorted(self.oov.items(), key=operator.itemgetter(1),
                                 reverse=True))
        #logging.debug('Evaluating predictions..')
        target = f'verb{cols_suff}' # or 'landmark'
        def is_good(series):
            return series[target] in series['predicted']
        target_df[f'good'] = target_df.apply(is_good, axis=1)
        n_good = target_df[f'good'].sum()
        # majority baseline = GS11 verb: 20 landmark: 10, KS13 verb1: 5 verb2: 7
        if n_good > 5:
            logging.info(f'{target}\t{weight}\t{rank}\t{n_good}')

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(levelname)-8s [%(lineno)d] %(message)s')
    evalor = VerbTensorEvaluator()
    evalor.test_sim(read_ks().reset_index(), mode_to_test='svo')
    #evalor.predict_verb(read_gs().reset_index())
    #test_sim(read_SimVerb().reset_index(), mode_to_test='ROOT')
    #test_sim(read_SimLex().reset_index(), mode_to_test='nsubj')
    #test_sim(read_SimLex().reset_index(), mode_to_test='dobj')
