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
logging.basicConfig(level=logging.DEBUG, format='%(levelname)-8s [%(lineno)d] %(message)s')



def select_from_conll(part=''):
    deps_wanted = set(['nsubj', 'ROOT', 'dobj', 'punct'])
    freq = defaultdict(int)
    for filen in glob.glob('/mnt/permanent/Language/English/Crawl/DepCC/corpus/parsed/part-m-*{}.gz'.format(part)):
        logging.info(filen)
        with gzip.open(filen, mode='rt', encoding="utf-8") as infile:
            triple = {}
            sentence_is_clean = True
            for i, line in enumerate(infile):
                if i and not i % 10000000:
                    logging.debug((i, len(freq)))
                line = line.strip()
                if line.startswith('#'):
                    # Finishes sentence and inits next.
                    if sentence_is_clean and triple.keys() == deps_wanted:
                        freq[(triple['nsubj'], triple['ROOT'], triple['dobj'])] += 1
                    sentence = []
                    triple = {}
                    sentence_is_clean = True
                    continue
                elif not line:
                    continue
                elif not sentence_is_clean:
                    continue
                id_, form, lemma, upostag, xpostag, feats, head, deprel, deps, ner = line.split('\t')
                sentence.append(form)
                if int(head) == 1:
                    if deprel in deps_wanted:
                        if deprel in triple:
                            sentence_is_clean = False
                        else:
                            triple[deprel] = lemma
                    else:
                        sentence_is_clean = False
        df = pd.DataFrame.from_records(
            [triple + tuple([count]) for (triple, count) in freq.items()], 
            columns=['nsubj', 'ROOT', 'dobj', 'freq']) 
        df = df.sort_values('freq', ascending=False) 
        df.to_pickle('/mnt/permanent/home/makrai/project/verb-tensor/just_svo/dataframe/depCC/freq{}.pkl'.format(part))
    return df


if __name__ == '__main__':
    select_from_conll(part=sys.argv[1])


# * conllu expects indexing from 1.
# * pyconll gives error parsing "None" fields
# * depedit: what can it do?
