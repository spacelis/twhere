#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: runtest.py
Author: SpaceLis
Changes:
    0.0.1 The first version
Description:
    running the experiments
"""
__version__ = '0.0.1'

import sys
import logging
logging.basicConfig(format='%(asctime)s %(name)s [%(levelname)s] %(message)s', level=logging.INFO)
LOGGER = logging.getLogger(__name__)
from twhere import run_experiment, PredictingMajority, PredictingLast, MarkovChainModel, ColfilterModel
import os
from config import *


def coreloop(poicol, model, resdir, name):
    """ coreloop is looping within the four cities
    """
    for city, city_id in zip(['NY', 'CH', 'LA', 'SF'],
                             ['27485069891a7938', '1d9a5370a355ab0c', '3b77caf94bfc81fe', '5a110d312052166f']):
        run_experiment(city_id, poicol, model, open(os.path.join(resdir, '%s_%s.res' % (city, name)), 'w'))


def experimentColfilter(poicol, resdir):
    """docstring for experimentColfilter
    """
    sigmahours = [4., 2., 1., 0.5, 0.25]
    for simnum in [100, 50, 20, 10, 5]:
        VECTORDB_PARAM['simnum'] = simnum
        for sigma, sigmahour in zip(map(lambda x: x * 3600., sigmahours), sigmahours):
            VECTORIZOR_PARAM['params'] = (sigma, )
            coreloop(poicol, ColfilterModel, resdir, 'n%03d_s%.1gh' % (simnum, sigmahour))


def experimentMarkovModel(poicol, resdir):
    """docstring for experimentColfilter
    """
    coreloop(poicol, MarkovChainModel, resdir, 'mm')


if __name__ == '__main__':
    LOGGER.debug('DEBUG is enabled')
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (2.5 * 1024 * 1024 * 1024L, -1L))
    except:
        LOGGER.warn('Failed set resource limits.')

    resdir = sys.argv[1]
    if not os.path.isdir(resdir):
        os.mkdir(resdir)
    #experimentColfilter('category', resdir)
    experimentMarkovModel('category', resdir)
    #import profile
    #profile.run('experiment()')
