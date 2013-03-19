#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: runtest.py
Author: SpaceLis
Changes:
    0.1.0 improved logging
    0.0.2 remove parallel as not supported on servers
    0.0.1 The first version
Description:
    running the experiments
"""
__version__ = '0.1.0'

import sys
import logging
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
# EXPERIMENTS
#
def experimentColfilter(poicol, resdir, city_idx=None):
    """docstring for experimentColfilter
    """
    conf = Configuration()
    conf['expr.target'] = poicol
    conf['expr.model'] = 'ColfilterModel'
    conf['cf.similarity'] = 'SparseCosineSimilarity'
    conf['vec.isaccum'] = True
    conf['vec.normalized'] = True
    sigmahours = [1., 4., 0.25, 12., 24.]
    for simnum in [100, 50, 5]:
        conf['cf.simnum'] = simnum
        for sigma, sigmahour in zip(map(lambda x: x * 3600., sigmahours), sigmahours):
            conf['vec.kernel.params'] = (sigma, )
            if city_idx is None:
                cityloop(conf, resdir, 'n%03d_s%6.2gh' % (simnum, sigmahour))
            else:
                run_experiment(conf, resdir, 'n%03d_s%6.2gh' % (simnum, sigmahour), city_idx)


def experimentMarkovModel(poicol, resdir, city_idx=None):
    """docstring for experimentColfilter
    """
    conf = Configuration()
    conf['expr.target'] = poicol
    conf['expr.model'] = 'MarkovChainModel'
    if city_idx is None:
        cityloop(conf, resdir, 'mm')
    else:
        run_experiment(conf, resdir, 'mm', city_idx)


def experimentPredictingMajority(poicol, resdir, city_idx=None):
    """docstring for experimentColfilter
    """
    conf = Configuration()
    conf['expr.target'] = poicol
    conf['expr.model'] = 'PredictingMajority'
    if city_idx is None:
        cityloop(conf, resdir, 'pm')
    else:
        run_experiment(conf, resdir, 'pm', city_idx)


def experimentPredictingTimeMajority(poicol, resdir, city_idx=None):
    """docstring for experimentColfilter
    """
    conf = Configuration()
    conf['expr.target'] = poicol
    conf['expr.model'] = 'PredictingTimeMajority'
    if city_idx is None:
        cityloop(conf, resdir, 'ptm')
    else:
        run_experiment(conf, resdir, 'ptm', city_idx)


def experimentPredictingLast(poicol, resdir, city_idx=None):
    """docstring for experimentColfilter
    """
    conf = Configuration()
    conf['expr.target'] = poicol
    conf['expr.model'] = 'PredictingLast'
    if city_idx is None:
        cityloop(conf, resdir, 'pl')
    else:
        run_experiment(conf, resdir, 'pl', city_idx)


def mockTrainTextExport(poicol, resdir, city_idx=None):
    """ export the training trails and testing instance for inspection
    """
    conf = Configuration()
    conf['expr.target'] = poicol
    conf['expr.model'] = 'mockTrainTextExport'
    if city_idx is None:
        cityloop(conf, resdir, 'tt')
    else:
        run_experiment(conf, resdir, 'tt', city_idx)


def setup_logging(filename, default_path='logging.conf.yaml', default_level=logging.INFO, env_key='LOG_CFG'):
    """ Setup logging
    """
    import yaml
    import logging.config
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            logconf = yaml.load(f.read())
            logconf['handlers']['info_file_handler']['filename'] = filename
        logging.config.dictConfig(logconf)
    else:
        logging.basicConfig(level=default_level)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print >> sys.stderr, 'No folder provided for results'
        print >> sys.stderr, 'Usage: runtest.py <dir>'
        sys.exit(-1)
    resdir = sys.argv[1]
    if not os.path.isdir(resdir):
        os.mkdir(resdir)
    setup_logging(sys.argv[1] + '/log')
    LOGGER = logging.getLogger(__name__)
    LOGGER.debug('DEBUG is enabled')
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (2.5 * 1024 * 1024 * 1024L, -1L))
    except:
        LOGGER.warn('Failed set resource limits.')

    city_idx = None if len(sys.argv) < 3 else int(sys.argv[2])
    #experimentPredictingLast('base', resdir, city_idx)
    #experimentMarkovModel('base', resdir, city_idx)
    #experimentPredictingMajority('base', resdir, city_idx)
    #experimentPredictingTimeMajority('base', resdir, city_idx)
    experimentColfilter('base', resdir, city_idx)
    #mockTrainTextExport('base', resdir, city_idx)
