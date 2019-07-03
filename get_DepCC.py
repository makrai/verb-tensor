#!/usr/bin/env python
# coding: utf-8


from collections import defaultdict
import glob
import gzip
import os
import pandas as pd
import pickle

import pandas as pd

import logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)-8s [%(lineno)d] %(message)s')


def select_from_conll():
    deps_wanted = ['nsubj', 'ROOT', 'dobj']
    freq = defaultdict(int)
    for filen in glob.glob('/mnt/permanent/Language/English/Crawl/DepCC/corpus/parsed/part-m-*00.gz'):
        logging.info(filen)
        with gzip.open(filen, mode='rt', encoding="utf-8") as infile:
            triple = {}
            argument_clash = False
            for i, line in enumerate(infile):
                if not i % 10000000:
                    logging.debug((i, len(freq)))
                line = line.strip()
                if line.startswith('#'):
                    for dep in deps_wanted:
                        if dep not in triple:
                            break
                    else:
                        if not argument_clash:
                            freq[(triple['nsubj'], triple['ROOT'], triple['dobj'])] += 1
                    sentence = []
                    triple = {}
                    argument_clash = False
                    continue
                if not line:
                    continue
                id_, form, lemma, upostag, xpostag, feats, head, deprel, deps, ner = line.split('\t')
                sentence.append(form)
                if deprel in deps_wanted and int(head)==1:
                    if deprel in triple:
                        #logging.warning('Multiple {}s: {}\n{}'.format(deprel, triple, ' '.join(sentence)))
                        argument_clash = True
                    triple[deprel] = lemma
                #logging.debug((triple, argument_clash, form))
    return freq


freq = select_from_conll() 
df = pd.DataFrame.from_records([triple+tuple([count]) for (triple, count) in freq.items()], 
                          columns=['nsubj', 'ROOT', 'dobj', 'freq']) 
df = df.sort_values('freq', ascending=False) 
df.to_csv('/mnt/store/home/makrai/project/verb-tensor/depCC-00.tsv', sep='\t', index=False)


# * conllu expects indexing from 1.
# * pyconll gives error parsing "None" fields
# * depedit: what can it do?
