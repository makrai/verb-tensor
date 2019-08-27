#!/usr/bin/env python
# coding: utf-8


from collections import defaultdict
import glob
import gzip
import pandas as pd
import pickle
import sys

import logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)-8s [%(lineno)d] %(message)s')


def select_from_conll(part=''):
    """
    Collects frequencies of subj, verb, dobj triples from depCC.
    Sencentes including top dependencies other than these (and punct) are
    disregarded.
    """
    columns = ['id', 'form', 'lemma', 'upos', 'xpos', 'feats', 'head_',
               'deprel', 'deps', 'ner']
    deps_wanted = set(['nsubj', 'ROOT', 'dobj', 'punct'])
    freq = defaultdict(int)
    for filen in glob.glob('/mnt/permanent/Language/English/Crawl/DepCC/corpus/parsed/part-m-*{}.gz'.format(part)):
        logging.info(filen)
        with gzip.open(filen, mode='rt', encoding="utf-8") as infile:
            lines = []
            triple = {}
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
                    pred_i = df[df.deprel == 'ROOT'].id.values[0]
                    top_df = df[df.head_ == pred_i]
                    if (set(top_df.deprel) == deps_wanted and
                            len(top_df.deprel) == 4):
                        for _, series in top_df.iterrows():
                            triple[series.deprel] = series.lemma 
                        freq[(triple['nsubj'], triple['ROOT'], triple['dobj'])] += 1
                        n_sents += 1
                        if not n_sents % 1000: 
                            logging.debug((
                                i, 
                                sorted(freq.items(), 
                                       key=lambda item: -item[1])[:4]))
                                #triple))
                    lines = []
                    triple = {}
        df = pd.DataFrame.from_records(
            [triple + tuple([count]) for (triple, count) in freq.items()],
            columns=['nsubj', 'ROOT', 'dobj', 'freq'])
        df = df.sort_values('freq', ascending=False)
        df.to_pickle('/mnt/permanent/home/makrai/project/verb-tensor/top_level/dataframe/freq{}.pkl'.format(part))
    return df


if __name__ == '__main__':
    if len(sys.argv) > 1:
        select_from_conll(part=sys.argv[1])
    else:
        select_from_conll()


# * conllu expects indexing from 1.
# * pyconll gives error parsing "None" fields
# * depedit: what can it do?
