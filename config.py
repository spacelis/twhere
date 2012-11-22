#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: config.py
Author: SpaceLis
Changes:
    0.0.1 The first version
Description:
    This is an example of config.py which will be used in twhere.py
"""
__version__ = '0.0.1'


import site
site.addsitedir('../model')

from model.colfilter import CosineSimilarity
from model.colfilter import HistoryDamper
from model.colfilter import LinearCombination
from model.colfilter import gaussian_pdf
from model.colfilter import exponent_pdf

HISTDAMP_PARAM = {"veclen": 100,
                  "interval": (0., 24 * 3600.),
                  "damp_func": exponent_pdf,
                  "params": {'l': 1 / 3600.}}

VECTORDB_PARAM = {"simnum": 20,
                  "similarity": CosineSimilarity(HistoryDamper(**HISTDAMP_PARAM)),
                  "aggregator": LinearCombination()}

VECTORIZOR_PARAM = {"veclen": 100,
                    "interval": (0., 24 * 3600.),
                    "kernel": gaussian_pdf,
                    "params": (3600.,),
                    "isaccum": False,
                    "normalized": False}

if __name__ == '__main__':
    raise Exception('Should run experiment.py')
