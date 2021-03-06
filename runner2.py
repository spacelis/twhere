#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: runner2.py
Author: SpaceLis
Changes:
    0.0.1 The first version
Description:
    Running experiments by giving a delta configuration
"""
__version__ = '0.0.1'

import sys
import os
import traceback
import json
import logging
import logging.config
import argparse
from StringIO import StringIO
from multiprocessing import cpu_count
from multiprocessing import RLock
from multiprocessing import Pool
from twhere.exprmodels import experiment
from twhere.config import Configuration

try:
    import affinity
    affinity.set_process_affinity_mask(os.getpid(), (1 << cpu_count()) - 1)
except:  # pylint: disable-msg=W0702
    print >>sys.stderr, 'WARN: Fail on setting CPU affinity, check cpu loads!'

CITY = dict(zip(['NY', 'CH', 'LA', 'SF'],
                ['27485069891a7938',
                 '1d9a5370a355ab0c',
                 '3b77caf94bfc81fe',
                 '5a110d312052166f']))

LOGGING_CONF = {'version': 1,
                'formatters': {
                'simple': {'format':
                           "%(asctime)s %(process)d %(name)s "
                           "[%(levelname)s] %(message)s"}
                },
                'handlers': {
                    'console': {'class': 'logging.StreamHandler',
                                'level': 'INFO',
                                'formatter': 'simple',
                                'stream': 'ext://sys.stderr'}
                },

                'root': {
                    'level': 'INFO',
                    'handlers': ['console', ]
                }
                }

OUTPUT_LOCK = RLock()


def worker(lconf):
    """ A worker function for wrapping prepare_and_run() with
        CPU affinity assignment.
    """
    try:
        prepare_and_run(lconf)
        with OUTPUT_LOCK:
            print '[SUCCEEDED]', lconf
    except Exception as e:
        exc_buffer = StringIO()
        traceback.print_exc(file=exc_buffer)
        logging.error('Uncaught exception in worker process:\n%s',
                      exc_buffer.getvalue())
        raise e


def pooling(lconf, poolsize=10):
    """ Running the list of conf in a multiprocess pool
    """
    pool = Pool(poolsize)
    pool.map(worker, lconf)


def prepare_and_run(deltaconf):
    """ Prepare the configuration and run experiments
    """
    conf = Configuration()
    conf.update(deltaconf)
    if conf['expr.city.id'] is None:
        conf['expr.city.id'] = CITY[conf['expr.city.name']]
    experiment(conf)


def setup_logging(logconf):
    """ Setup logging
    """
    logging.config.dictConfig(logconf)


def parse_parameter():
    """ Parse the argument
    """
    parser = argparse.ArgumentParser(description='Running Trail Prediction')
    parser.add_argument(
        '-f', dest='conffile',
        action='store',
        metavar='FILE',
        default=None,
        help='Running with delta configuration from the FILE')
    parser.add_argument(
        '-s',
        dest='confstr',
        action='store',
        metavar='JSON',
        default=None,
        help='Running with the delta configuration from the json string')
    parser.add_argument(
        '-p', dest='pooled',
        action='store',
        nargs=2,
        metavar=('POOLSIZE', 'FILE'),
        default=None,
        help='Running a list of configuration in a multiprocess pool')
    parser.add_argument(
        '--loglevel', dest='log_level',
        action='store',
        metavar='INFO',
        default='INFO',
        help='The level of log output')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    appargs = parse_parameter()
    if appargs.log_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
        LOGGING_CONF['root']['level'] = appargs.log_level
    setup_logging(LOGGING_CONF)
    LOGGER = logging.getLogger(__name__)
    LOGGER.debug('DEBUG is enabled')
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (1500 * 1024 * 1024L, -1L))
    except ValueError as err:
        LOGGER.warn('Failed set resource limits. Because {0}'.
                    format(err.message))

    if appargs.pooled is not None:
        with open(appargs.pooled[1]) as fconf:
            pooling([json.loads(l) for l in fconf], int(appargs.pooled[0]))
        exit(0)
    if appargs.conffile is not None:
        with open(appargs.conffile) as fconf:
            dconf = json.loads(fconf.read())
    if appargs.confstr is not None:
        dconf = json.loads(appargs.confstr)
    prepare_and_run(dconf)
