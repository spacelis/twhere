#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: runtest.py
Author: SpaceLis
Changes:
    0.0.2 remove parallel as not supported on servers
    0.0.1 The first version
Description:
    running the experiments
"""
__version__ = '0.0.2'

import sys
import logging
logging.basicConfig(format='%(asctime)s %(name)s [%(levelname)s] %(message)s', level=logging.INFO)
LOGGER = logging.getLogger(__name__)
import multiprocessing
from twhere import run_experiment, PredictingMajority, PredictingTimeMajority, PredictingLast, MarkovChainModel, ColfilterModel, ColfilterHistoryModel, generate_testing_reference
import os
import config
from model.colfilter import CosineSimilarity, HistoryDamper, uniform_pdf


# -------------------------------------------------
# The core loop
# -------------------------------------------------

CITY = zip(['NY', 'CH', 'LA', 'SF'],
           ['27485069891a7938', '1d9a5370a355ab0c', '3b77caf94bfc81fe', '5a110d312052166f'])


def core(poicol, model, resdir, name, city_idx):
    """ run experiment in one city
    """
    run_experiment(CITY[city_idx][1], poicol, model, os.path.join(resdir, '%s_%s.res' % (CITY[city_idx][0], name)))


def coreloop(poicol, model, resdir, name):
    """ coreloop is looping within the four cities
    """
    for city_idx in range(len(CITY)):
        core(poicol, model, resdir, name, city_idx)


# -------------------------------------------------
# Parallelism of coreloop
#
# Do not work on servers
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


# -------------------------------------------------
# EPERIMENTS
#
def experimentColfilter(poicol, resdir, city_idx=None):
    """docstring for experimentColfilter
    """
    config.reset_config()
    sigmahours = [0.25, 1., 4., 12., 24.]
    for simnum in [100, 50, 5]:
        config.VECTORDB_PARAM['simnum'] = simnum
        config.VECTORIZOR_PARAM['isaccum'] = True
        config.VECTORIZOR_PARAM['normalized'] = True
        for sigma, sigmahour in zip(map(lambda x: x * 3600., sigmahours), sigmahours):
            config.VECTORIZOR_PARAM['params'] = (sigma, )
            config.VECTORIZOR_PARAM['kernel'] = uniform_pdf
            if city_idx is None:
                coreloop(poicol, ColfilterModel, resdir, 'n%03d_s%6.2gh' % (simnum, sigmahour))
            else:
                core(poicol, ColfilterModel, resdir, 'n%03d_s%6.2gh' % (simnum, sigmahour), city_idx)


def generate_refs(poicol, resdir, city_idx=None):
    for city_idx in range(len(CITY)):
        generate_testing_reference(CITY[city_idx][1], poicol, None, os.path.join(resdir, '%s.res' % (CITY[city_idx][0], )))


def experimentColfilterHistory(poicol, resdir, city_idx=None):
    """docstring for experimentColfilter
    """
    config.reset_config()
    sigmahours = [4., 24., 1.]
    for simnum in [100, 50, 5]:
        config.VECTORDB_PARAM['simnum'] = simnum
        config.VECTORIZOR_PARAM['isaccum'] = True
        config.VECTORIZOR_PARAM['normalized'] = True
        for sigma, sigmahour in zip(map(lambda x: x * 3600., sigmahours), sigmahours):
            config.VECTORIZOR_PARAM['params'] = (sigma, )
            if city_idx is None:
                coreloop(poicol, ColfilterHistoryModel, resdir, 'n%03d_s%6.2gh' % (simnum, sigmahour))
            else:
                core(poicol, ColfilterHistoryModel, resdir, 'n%03d_s%6.2gh' % (simnum, sigmahour), city_idx)


def experimentColfilterHistoryDiscounting(poicol, resdir):
    """docstring for experimentColfilterHistoryDiscounting
    """
    config.reset_config()
    sigmahours = [4., 1., 0.25]
    for simnum in [100, 20, 5]:
        config.VECTORDB_PARAM['simnum'] = simnum
        for sigma, sigmahour in zip(map(lambda x: x * 3600., sigmahours), sigmahours):
            for l in [14400, 7200, 3600, 1800, 900]:
                config.HISTDAMP_PARAM['params'] = {'l': 1. / l, }
                config.VECTORDB_PARAM['similarity'] = CosineSimilarity([HistoryDamper(**config.HISTDAMP_PARAM), ])
                config.VECTORIZOR_PARAM['params'] = (sigma, )
                coreloop(poicol, ColfilterModel, resdir, 'n%03d_s%6.2gh_d%05d' % (simnum, sigmahour, l))


def experimentMarkovModel(poicol, resdir):
    """docstring for experimentColfilter
    """
    coreloop(poicol, MarkovChainModel, resdir, 'mm')


def experimentPredictingMajority(poicol, resdir):
    """docstring for experimentColfilter
    """
    coreloop(poicol, PredictingMajority, resdir, 'pm')


def experimentPredictingTimeMajority(poicol, resdir):
    """docstring for experimentColfilter
    """
    coreloop(poicol, PredictingTimeMajority, resdir, 'ptm')


def experimentPredictingLast(poicol, resdir):
    """docstring for experimentColfilter
    """
    coreloop(poicol, PredictingLast, resdir, 'pl')


if __name__ == '__main__':
    LOGGER.debug('DEBUG is enabled')
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (2.5 * 1024 * 1024 * 1024L, -1L))
    except:
        LOGGER.warn('Failed set resource limits.')

    if len(sys.argv) < 2:
        LOGGER.fatal('No folder provided for results')
        print >> sys.stderr, 'Usage: runtest.py <dir>'
        sys.exit(-1)
    resdir = sys.argv[1]
    if not os.path.isdir(resdir):
        os.mkdir(resdir)

    # -------------------------------------------------
    # Do NOT run experiments at one go, the parameters would confuse
    # -------------------------------------------------

    #experimentPredictingLast('category', resdir)
    #experimentMarkovModel('category', resdir)
    #experimentPredictingMajority('category', resdir)
    #experimentPredictingTimeMajority('category', resdir)
    #experimentColfilter('base', resdir, None if len(sys.argv) < 3 else int(sys.argv[2]))
    generate_refs('base', resdir, None if len(sys.argv) < 3 else int(sys.argv[2]))
    #experimentColfilterHistory('category', resdir, None if len(sys.argv) < 3 else int(sys.argv[2]))
    #experimentColfilterHistoryDiscounting('category', resdir)
    #import profile
    #profile.run('experiment()')
