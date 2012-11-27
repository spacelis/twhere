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
import multiprocessing
from twhere import run_experiment, PredictingMajority, PredictingLast, MarkovChainModel, ColfilterModel
import os
from config import VECTORIZOR_PARAM, VECTORDB_PARAM


def coreloop(poicol, model, resdir, name):
    """ coreloop is looping within the four cities
    """
    for city, city_id in zip(['NY', 'CH', 'LA', 'SF'],
                             ['27485069891a7938', '1d9a5370a355ab0c', '3b77caf94bfc81fe', '5a110d312052166f']):
        run_experiment(city_id, poicol, model, os.path.join(resdir, '%s_%s.res' % (city, name)))


def call_experiment(args):
    """ one parameter interface for run_experiment
    """
    run_experiment(*args)


def coreloop_parallel(poicol, model, resdir, name):
    """ coreloop_parallel is looping within the four cities
    """
    paralist = list()
    for city, city_id in zip(['NY', 'CH', 'LA', 'SF'],
                             ['27485069891a7938', '1d9a5370a355ab0c', '3b77caf94bfc81fe', '5a110d312052166f']):
        paralist.append((city_id, poicol, model, os.path.join(resdir, '%s_%s.res' % (city, name))))
    pool = multiprocessing.Pool(4)
    pool.map(call_experiment, paralist)


def experimentColfilter(poicol, resdir):
    """docstring for experimentColfilter
    """
    sigmahours = [4., 2., 1., 0.5, 0.25]
    for simnum in [100, 50, 20, 10, 5]:
        VECTORDB_PARAM['simnum'] = simnum
        for sigma, sigmahour in zip(map(lambda x: x * 3600., sigmahours), sigmahours):
            VECTORIZOR_PARAM['params'] = (sigma, )
            coreloop_parallel(poicol, ColfilterModel, resdir, 'n%03d_s%.1gh' % (simnum, sigmahour))


def experimentMarkovModel(poicol, resdir):
    """docstring for experimentColfilter
    """
    coreloop_parallel(poicol, MarkovChainModel, resdir, 'mm')


def experimentPredictingMajority(poicol, resdir):
    """docstring for experimentColfilter
    """
    coreloop_parallel(poicol, PredictingMajority, resdir, 'pm')


def experimentPredictingLast(poicol, resdir):
    """docstring for experimentColfilter
    """
    coreloop_parallel(poicol, PredictingLast, resdir, 'pl')


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
    experimentPredictingLast('category', resdir)
    experimentMarkovModel('category', resdir)
    experimentPredictingMajority('category', resdir)
    experimentColfilter('category', resdir)
    #import profile
    #profile.run('experiment()')
