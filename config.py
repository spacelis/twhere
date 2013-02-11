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
from model.colfilter import LinearCombination
from model.colfilter import gaussian_pdf
from model.colfilter import exponent_pdf

HISTDAMP_PARAM_DEFAULT = {"veclen": 100,
                          "interval": (0., 24 * 3600.),
                          "damp_func": exponent_pdf,
                          "params": {'l': 1 / 3600.}}

VECTORDB_PARAM_DEFAULT = {"simnum": 20,
                          "similarity": CosineSimilarity([]),
                          "aggregator": LinearCombination()}

VECTORIZOR_PARAM_DEFAULT = {"veclen": 100,
                            "interval": (0., 24 * 3600.),
                            "kernel": gaussian_pdf,
                            "params": (3600.,),
                            "isaccum": False,
                            "normalized": False}

HISTDAMP_PARAM = dict(HISTDAMP_PARAM_DEFAULT)

VECTORDB_PARAM = dict(VECTORDB_PARAM_DEFAULT)

VECTORIZOR_PARAM = dict(VECTORIZOR_PARAM_DEFAULT)


def reset_config():
    """ reset the configuration
    """
    global HISTDAMP_PARAM, VECTORDB_PARAM, VECTORIZOR_PARAM
    HISTDAMP_PARAM = dict(HISTDAMP_PARAM_DEFAULT)
    VECTORDB_PARAM = dict(VECTORDB_PARAM_DEFAULT)
    VECTORIZOR_PARAM = dict(VECTORIZOR_PARAM_DEFAULT)


if __name__ == '__main__':
    raise Exception('Should run experiment.py')
