#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8


# In[ ]:


from collections import defaultdict
import glob
import itertools
import os
import pandas as pd
import pickle
import lzma

from conllu import parse, parse_incr
import numpy as np

import logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)-8s [%(lineno)d] %(message)s')


# In[ ]:


proj_dir = '/mnt/store/home/makrai/project/verb-tensor/'

def get_umbc_dict():
    dict_filen = os.path.join(proj_dir, 'umbc_freq.pkl')
    if os.path.exists(dict_filen):
        return pickle.load(open(dict_filen, mode='rb'))
    umbc_dir = '/mnt/store/home/makrai/data/language/english/corp/umbc_WebBase/English/'
    freq = defaultdict(int)
    for filen in glob.glob(os.path.join(umbc_dir, 'en-common_crawl-*.conllu.xz')):
        logging.info(filen)
        for i, sentence in  enumerate(parse_incr(lzma.open(os.path.join(
                umbc_dir, filen), mode='rt', encoding="utf-8"))):
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
        pickle.dump(freq, open(dict_filen, mode='wb'))
    return freq


# In[ ]:


def get_umbc_df():
    freq = get_umbc_dict()
    freq_df = pd.DataFrame.from_records(list(freq.items()), columns=['svo', 'freq'])
    freq_df[['subj', 'verb', 'obj']] = pd.DataFrame(freq_df.svo.tolist(),
            index=freq_df.index)
    del freq_df['svo']
    df.sort_values('freq', ascending=False).to_csv(os.path.join(proj_dir, 'umbc_freq.csv'), sep='\t', index=False)
    return freq_df


# In[ ]:


df = get_umbc_df()


# In[ ]:


df[(df.subj != '') & (df.obj != '')].sort_values('freq', ascending=False).head()


# In[ ]:


df = get_umbc_df()

