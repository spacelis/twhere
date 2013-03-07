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
from twhere.exprmodels import experiment

import os
from twhere.config import Configuration


# -------------------------------------------------
# The expriment loop
# -------------------------------------------------

CITY = zip(['NY', 'CH', 'LA', 'SF'],
           ['27485069891a7938', '1d9a5370a355ab0c', '3b77caf94bfc81fe', '5a110d312052166f'])


def run_experiment(conf, resdir, name, city_idx):
    """ run experiment in one city
    """
    conf['expr.city.name'] = CITY[city_idx][0]
    conf['expr.city.id'] = CITY[city_idx][1]
    conf['expr.output'] = os.path.join(resdir, '%s_%s.res' % (conf['expr.city.name'], name))
    experiment(conf)


def cityloop(conf, resdir, name):
    """ cityloop is looping within the four cities
    """
    for city_idx in range(len(CITY)):
        run_experiment(conf, resdir, name, city_idx)


# -------------------------------------------------
# EPERIMENTS
#
def experimentColfilter(poicol, resdir, city_idx=None):
    """docstring for experimentColfilter
    """
    conf = Configuration()
    conf['expr.target'] = 'base'
    conf['expr.model'] = 'ColfilterModel'
    conf['vec.isaccum'] = True
    conf['vec.normalized'] = True
    sigmahours = [0.25, 1., 4., 12., 24.]
    for simnum in [100, 50, 5]:
        conf['cf.simnum'] = simnum
        for sigma, sigmahour in zip(map(lambda x: x * 3600., sigmahours), sigmahours):
            conf['vec.kernel.params'] = (sigma, )
            if city_idx is None:
                cityloop(conf, resdir, 'n%03d_s%6.2gh' % (simnum, sigmahour))
            else:
                run_experiment(conf, resdir, 'n%03d_s%6.2gh' % (simnum, sigmahour), city_idx)


def experimentMarkovModel(poicol, resdir):
    """docstring for experimentColfilter
    """
    conf = Configuration()
    conf['expr.model'] = 'MarkovChainModel'
    cityloop(conf, resdir, 'mm')


def experimentPredictingMajority(poicol, resdir):
    """docstring for experimentColfilter
    """
    conf = Configuration()
    conf['expr.model'] = 'PredictingMajority'
    cityloop(conf, resdir, 'pm')


def experimentPredictingTimeMajority(poicol, resdir):
    """docstring for experimentColfilter
    """
    conf = Configuration()
    conf['expr.model'] = 'PredictingTimeMajority'
    cityloop(conf, resdir, 'ptm')


def experimentPredictingLast(poicol, resdir):
    """docstring for experimentColfilter
    """
    conf = Configuration()
    conf['expr.model'] = 'PredictingLast'
    cityloop(conf, resdir, 'pl')


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
    experimentColfilter('base', resdir, None if len(sys.argv) < 3 else int(sys.argv[2]))
    #generate_refs('base', resdir, None if len(sys.argv) < 3 else int(sys.argv[2]))
    #experimentColfilterHistory('category', resdir, None if len(sys.argv) < 3 else int(sys.argv[2]))
    #experimentColfilterHistoryDiscounting('category', resdir)
    #import profile
    #profile.run('experiment()')
