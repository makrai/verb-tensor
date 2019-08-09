#!/usr/bin/env python
# coding: utf-8


from collections import defaultdict
import glob
import gzip
import pandas as pd

import logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)-8s [%(lineno)d] %(message)s') 


def top_rel_stat(part='0000'):
    freq = defaultdict(int)
    for filen in glob.glob('/mnt/permanent/Language/English/Crawl/DepCC/corpus/parsed/part-m-*{}.gz'.format(part)):
        logging.info(filen)
        with gzip.open(filen, mode='rt', encoding="utf-8") as infile:
            top_rels = []
            for i, line in enumerate(infile):
                if not i % 1000000:
                    logging.debug((i, len(freq)))
                line = line.strip()
                if line.startswith('#'):
                    freq[str(top_rels)] += 1
                    sentence = []
                    top_rels = []
                    continue
                if not line:
                    continue
                id_, form, lemma, upostag, xpostag, feats, head, deprel, deps, ner = line.split('\t')
                sentence.append(form)
                if int(head) == 1:
                    top_rels.append(deprel) 
    return freq
    df = pd.DataFrame.from_records([top_rels+tuple([count]) 
                                    for (top_rels, count) in freq.items()], 
                                   columns=['nsubj', 'ROOT', 'dobj', 'freq']) 
    return df.sort_values('freq', ascending=False) 
