#!/usr/bin/env python
# coding: utf-8



import logging
import sys

from decomp_pmi import VerbTensor

logging.basicConfig(level=logging.DEBUG, format='%(levelname)-8s [%(lineno)d] %(message)s')      


if len(sys.argv) > 1:
    vt = VerbTensor(input_part=sys.argv[1])
else:
    vt = VerbTensor('')
vt.append_pmi()
