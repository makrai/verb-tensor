#!/usr/bin/env python
# coding: utf-8

# In[23]:


from decomp_pmi import VerbTensor
import logging
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic('pylab', 'inline')
pylab.rcParams['figure.figsize'] = (20, 10)

logging.basicConfig(level=logging.DEBUG, format='%(levelname)-8s [%(lineno)d] %(message)s')


# In[2]:


vt = VerbTensor('0to4')


# In[3]:


df = vt.append_pmi(positive=False)


# In[4]:


desc = df.describe(percentiles=[])
by_p = df.sort_values('npmi', ascending=False)
by_i = df.sort_values('niact', ascending=False)


# In[11]:


freqs_and_pmis = ['freq', 'freq_nsubj', 'freq_ROOT', 'freq_dobj',
       "freq_('nsubj', 'ROOT')", "freq_('nsubj', 'dobj')",
       "freq_('ROOT', 'dobj')", 'pmi', 'iact_info', 'npmi', 'niact']


# In[5]:


desc[['npmi', 'niact']].loc[['min', 'max']]


# In[25]:


df0 = df[df.freq >= 5]
plt.scatter(df0.npmi, df0.niact, s=.5)


# In[12]:


by_p[freqs_and_pmis].head()


# In[13]:


by_p[freqs_and_pmis].tail()


# In[14]:


by_i[freqs_and_pmis].head()


# In[15]:


by_i[freqs_and_pmis].tail()

