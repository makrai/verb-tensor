#!/usr/bin/env python
# coding: utf-8


from collections import defaultdict
from glob import glob
import gzip
import logging
import pandas as pd
import sys

import pyconll


logging.basicConfig(level=logging.DEBUG, format='%(levelname)-8s [%(lineno)d] %(message)s')


def depcc_to_conllu(filen_gz):
    """Fixes indexing and feats, and yields conllus of a sentence, one sentence each time."""
    with gzip.open(filen_gz, mode='rt') as infile:
        lines = []
        for line in infile:
            if line.startswith('# '):
                lines.append(line)
            elif line.strip():
                vals = line.split('\t')
                vals[0] = str(int(vals[0])+1)
                vals[5] = '_'
                line = '\t'.join(vals)
                lines.append(line)
            else:
                yield ''.join(lines)
                lines = []
        else:
            yield ''.join(lines)


def get_triples(input_part=9100):
    triples = []
    for filen in glob(f'/mnt/permanent/Language/English/Crawl/DepCC/corpus/parsed/part-m-*{input_part}.gz'):
        logging.info(filen)
        for sent_str in depcc_to_conllu(filen):
            train = pyconll.load_from_string(sent_str)
            sentence = train.pop() # sent_str is one sentence.

            triples_in_sent = defaultdict(dict)
            # triples = {id_of_root: {'nsubj': 'dog'}}

            # Collecting the arguments..
            for token in sentence:
                if token.deprel in ['nsubj', 'dobj']:
                    triples_in_sent[token.head][token.deprel] = token.lemma

            # Collecting the verbs..
            for id_form_1 in triples_in_sent:
                triples_in_sent[id_form_1]['ROOT'] = sentence[int(id_form_1)].lemma


            # Appending full triples to the list..
            for triple in triples_in_sent.values():
                if 'dobj' in triple and 'nsubj' in triple:
                    triples.append(triple)

    df = pd.DataFrame(triples)
    ser = df.groupby(list(df.columns)).size().sort_values(ascending=False)
    ser.to_pickle(f'/mnt/permanent/home/makrai/project/verb-tensor/pyconll/dataframe/freq{input_part}.pkl')
    return ser


if __name__ == '__main__':
    if len(sys.argv) > 1:
        get_triples(input_part=sys.argv[1])
    else:
        get_triples()
