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


config = configparser.ConfigParser()
config.read('config.ini')
tensor_dir = config['DEFAULT']['ProjectDirectory']+'tensor/'
test_data_dir = '/mnt/permanent/Language/English/Data/'
verb_sim_data_dir = f'{test_data_dir}verb-similarity/Sadrzadeh/'


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
    def __init__(self, weight='pmi-sali', include_empty=True, cutoff=1000000,
                 non_negative=True, decomp_algo='parafac', rank=64,
                 normlz_vocb=False, lmbda=False, mode_to_test='svo'):
        self.non_negative = non_negative
        self.decomp_algo = decomp_algo
        self.weight = weight
        self.rank = rank
        self.cutoff = cutoff
        self.include_empty = include_empty
        self.mode_to_test = mode_to_test
        self.normlz_vocb = normlz_vocb
        self.lmbda = lmbda

    def get_cols(self):
        if self.mode_to_test == 'svo':
            # logging.info('Assuming df is Kartsaklis and Sadrzadeh Turk')
            self.query1_cols = list(enumerate(['subject1', 'verb1', 'object1']))
            self.query2_cols = list(enumerate(['subject2', 'verb2', 'object2']))
            self.sim_col = 'score'
        elif self.mode_to_test == 'ROOT':
            # logging.info('Assuming df is SimVerb')
            self.query1_cols = [(1, 'verb1')]
            self.query2_cols = [(1, 'verb2')]
            self.sim_col = 'sim'
        elif self.mode_to_test == 'nsubj':
            # logging.info('Assuming df is SimLex')
            self.query1_cols = [(0, 'word1')]
            self.query2_cols = [(0, 'word2')]
            self.sim_col = 'SimLex999'
        elif self.mode_to_test == 'dobj':
            # logging.info('Assuming df is SimLex')
            self.query1_cols = [(2, 'word1')]
            self.query2_cols = [(2, 'word2')]
            self.sim_col = 'SimLex999'
        else:
            raise Exception(
                'mode_to_test has to be eigther svo, nsubj, ROOT, or dobj')

    def load_embeddings(self):
        non_negative_str = 'nonneg' if self.non_negative else 'general'
        include_empty_str = 'optional' if self.include_empty else 'non-empty'
        basen = f'{non_negative_str}_{self.decomp_algo}_{self.weight}_{include_empty_str}_{self.cutoff}_{self.rank}.pkl'
        self.decomped_tns = pickle.load(open(os.path.join(tensor_dir, basen),
                                             mode='rb'))
        factors = self.decomped_tns.factors
        logging.debug(self.include_empty)
        _, self.index = pickle.load(open(os.path.join(
            tensor_dir,
            f'sparstensr_{self.weight}_{include_empty_str}_{self.cutoff}.pkl'), mode='rb'))
        if self.decomp_algo == 'parafac':
            factors = [factor.todense() for factor in factors]
        if self.lmbda:
            sq_lam = np.sqrt(np.apply_along_axis(np.linalg.norm, 0,
                                                 self.decomped_tns.weights.todense()))
            for mode_i in range(3):
                factors[mode_i] *= sq_lam
        if self.normlz_vocb and self.mode_to_test != 'svo':
            mode_i = self.query1_cols[0][0]
            factors[mode_i] /= np.apply_along_axis(
                np.linalg.norm, 1, factors[mode_i]).reshape((-1, 1))
        self.modes = ['nsubj', 'ROOT', 'dobj']
        self.oov = defaultdict(int)

        def lookup(word, mode_i=1):
            if word in self.index[self.modes[mode_i]]:
                return factors[mode_i][self.index[self.modes[mode_i]][word]]
            else:
                self.oov[word] += 1
                return np.zeros(self.rank)
        self.lookup = lookup

    def test_sim(self, task_df0):
        task_df = task_df0.copy()  # Subj and obj sim not to go in the same df
        self.get_cols()
        self.load_embeddings()
        for mode_i, query_w_col in self.query1_cols + self.query2_cols:
            task_df[f'{query_w_col}_v'] = task_df[query_w_col].apply(
                lambda word: self.lookup(word, mode_i=mode_i))
            task_df[f'{query_w_col}_known'] = task_df[query_w_col].apply(
                lambda word: word in self.index[self.modes[mode_i]])
        words_known = task_df[[f'{query_w_col[1]}_known'
                               for query_w_col in self.query1_cols + self.query2_cols]]
        known_word_ratio = words_known.all(axis='columns').mean()
        if self.mode_to_test == 'svo':
            for qwocs, svo_vc in [(self.query1_cols, 'svo1_v'),
                                  (self.query2_cols, 'svo2_v')]:
                qvecs = [f'{qwc[1]}_v' for qwc in qwocs]
                task_df[svo_vc] = task_df[qvecs].apply(np.concatenate, axis=1)
        if self.normlz_vocb and self.mode_to_test == 'svo':
            for svo_vc in ['svo1_v', 'svo2_v']:
                task_df[svo_vc] /= task_df[svo_vc].apply(np.linalg.norm)
        if self.mode_to_test == 'svo':
            query_v_cols = ['svo1_v', 'svo2_v']
        else:
            query_v_cols = [f'{item[0][1]}_v'
                            for item in [self.query1_cols, self.query2_cols]]

        def cell_dot_cell(series):
            return series[0].dot(series[1])
        target_col = 'tensor_sim'  # _{self.weight}_{self.cutoff}_{self.rank}'
        task_df[target_col] = task_df[query_v_cols].apply(cell_dot_cell, axis=1)
        top_oov = sorted(self.oov.items(), key=operator.itemgetter(1), reverse=True)[:5]
        logging.debug(f'OOV: {top_oov}')
        sim_corr = task_df.corr(method='spearman').loc[self.sim_col]
        logging.debug(sim_corr)
        return sim_corr[target_col], known_word_ratio

    def predict_verb(self, target_df, prec_at=1, logg_oov=False,
                     target_pref='verb', cols_suff='', majority_baseline=5):
        """
        Results are bellow the majority baseline.
        cols_suff: GS11 or Jenatton: '', KS13: 1 or 2
        majority baseline: GS11 verb: 20 landmark: 10, KS13 verb1: 5 verb2: 7
        target_pref: GS11: verb or landmark, KS13: verb
        """
        self.load_embeddings()
        # logging.debug('Making predictions..')
        low_verb_low = tl.tenalg.mode_dot(
                self.decomped_tns.core, self.decomped_tns.factors[1], mode=1)

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
        # logging.debug('Evaluating predictions..')
        target = f'{target_pref}{cols_suff}'

        def is_good(series):
            return series[target] in series['predicted']
        target_df[f'good'] = target_df.apply(is_good, axis=1)
        n_good = target_df[f'good'].sum()
        if n_good > 5:
            logging.info(f'{target}\t{self.weight}\t{self.rank}\t{self.include_empty}\t{self.cutoff}\t{n_good}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(levelname)-8s [%(lineno)d] %(message)s')
    evalor = VerbTensorEvaluator(weight='niact', include_empty=False,
                                 cutoff=1000000, rank=512)
    evalor.test_sim(read_ks().reset_index())
    # evalor.predict_verb(read_gs().reset_index())
