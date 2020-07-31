#!/usr/bin/env python
# coding: utf-8


from collections import defaultdict
import glob
import gzip
import os
import pandas as pd
import pickle
import sys

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)-8s [%(lineno)d] %(message)s')


def select_from_conll(part=''):
    """
    Collects frequencies of subj, verb, dobj triples from depCC,
    including sencentes with top dependencies other than these.
    """
    columns = ['id', 'form', 'lemma', 'upos', 'xpos', 'feats', 'head_',
               'deprel', 'deps', 'ner']
    deps_wanted = set(['nsubj', 'ROOT', 'dobj'])
    freq = defaultdict(int)
    depcc_dir = '/mnt/permanent/Language/English/Crawl/DepCC/corpus/parsed'
    for filen in sorted(glob.glob(os.path.join(depcc_dir,
                                               f'part-m-*{part}.gz'))):
        logging.info(filen)
        with gzip.open(filen, mode='rt', encoding="utf-8") as infile:
            lines = []
            triple = defaultdict(str)
            n_sents = 0
            for i, line in enumerate(infile):
                line = line.strip()
                if line.startswith('#'):
                    continue
                elif line:
                    lines.append(line.split('\t'))
                elif lines:
                    # Finishes sentence and inits next.
                    df = pd.DataFrame.from_records(lines, columns=columns)
                    main_verb = df[(df.deprel == 'ROOT') &
                                   (df.upos.str.startswith('VB'))]
                    if main_verb.shape[0] == 1:
                        pred_i = main_verb.id.values[0]
                    else:
                        if main_verb.shape[0] > 1:
                            logging.debug(f'More roots: \n{df}')
                        pred_i = '-1'
                    top_df = df[df.head_ == pred_i]
                    #if deps_wanted <= set(top_df.deprel):
                    for _, series in top_df.reset_index().iterrows():
                        triple[series.deprel] = series.lemma.lower()
                    if ('nsubj' in triple and 'dobj' in triple):
                        freq[(triple['nsubj'], triple['ROOT'], triple['dobj'])] += 1
                    n_sents += 1
                    if not n_sents % 100000:
                        logging.debug((
                            i,
                            sorted(freq.items(),
                                   key=lambda item: -item[1])[:3]))
                    lines = []
                    triple = defaultdict(str)
        df = pd.DataFrame.from_records(
            [triple + tuple([count]) for (triple, count) in freq.items()],
            columns=['nsubj', 'ROOT', 'dobj', 'freq']).set_index(
                ['nsubj', 'ROOT', 'dobj'])
        df = df.sort_values('freq', ascending=False)
        df.to_pickle(config['DEFAULT']['ProjectDirectory']+f'dataframe/freq{part}.pkl')
    return df


if __name__ == '__main__':
    if len(sys.argv) > 1:
        select_from_conll(part=sys.argv[1])
    else:
        select_from_conll()


# * conllu expects indexing from 1.
# * pyconll gives error parsing "None" fields
# * depedit: what can it do?
