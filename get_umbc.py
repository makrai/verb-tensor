#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import defaultdict
import itertools
import os
import pandas as pd
import pickle
import lzma

from conllu import parse, parse_incr
import numpy as np

import logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)-8s [%(lineno)d] %(message)s')


def get_umbc_dict():
    umbc_dir = '/mnt/store/home/makrai/data/language/english/corp/umbc_WebBase/English/'
    freq = defaultdict(int)
    for filen in os.listdir(umbc_dir):
        logging.info(filen)
        for i, sentence in  enumerate(parse_incr(lzma.open(os.path.join(umbc_dir, filen), mode='rt',
                                                           encoding="utf-8"))):
            if not i % 100000:
                logging.debug(i)
            root = sentence.to_tree()
            subj, obj = '', ''
            for child in root.children:
                if 'subj' in child.token['deprel']:
                    if subj:
                        #logging.warn('subj: {}'.format((subj, child.token['lemma'], sentence)))
                        continue
                    subj = child.token['lemma']
                elif child.token['deprel'] == 'obj':
                    if obj:
                        #logging.warn('obj: {}'.format((obj, child.token['lemma'], sentence)))
                        continue            
                    obj = child.token['lemma']
            #if bool(obj) and bool(subj):
            freq[(subj, root.token['lemma'], obj)] += 1
        #pickle.dump(freq, open('/mnt/store/home/makrai/project/verb-tensor/umbc_freq.pkl', mode='wb'))
    return freq


def get_umbc_df():
    freq = pickle.load(open('/mnt/store/home/makrai/project/verb-tensor/umbc_freq.pkl', mode='rb'))
    freq_df = pd.DataFrame.from_records(list(freq.items()), columns=['svo', 'freq'])
    freq_df[['subj', 'verb', 'obj']] = pd.DataFrame(freq_df.svo.tolist(), index=freq_df.index)                                                                                                                       
    del freq_df['svo']
    return freq_df
