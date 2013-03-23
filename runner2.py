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

import json
import logging
import logging.config
import argparse
from twhere.exprmodels import experiment
from twhere.config import Configuration

CITY = dict(zip(['NY', 'CH', 'LA', 'SF'],
           ['27485069891a7938', '1d9a5370a355ab0c', '3b77caf94bfc81fe', '5a110d312052166f']))

LOGGING_CONF = {'version': 1,
                'formatters': {
                    'simple': {'format': "%(asctime)s %(name)s [%(levelname)s] %(message)s"}
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


def setup_logging(logconf):
    """ Setup logging
    """
    logging.config.dictConfig(logconf)


def parse_parameter():
    """ Parse the argument
    """
    parser = argparse.ArgumentParser(description='Running Trail Prediction')
    parser.add_argument('-f', dest='conffile', action='store', metavar='FILE', default=None,
            help='Running with delta configuration from the FILE')
    parser.add_argument('-s', dest='confstr', action='store', metavar='JSON', default=None,
            help='Running with the delta configuration from the json string')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    setup_logging(LOGGING_CONF)
    LOGGER = logging.getLogger(__name__)
    LOGGER.debug('DEBUG is enabled')
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (800 * 1024 * 1024L, -1L))
    except ValueError as err:
        LOGGER.warn('Failed set resource limits. Because {0}'.format(err.message))

    appargs = parse_parameter()
    if appargs.conffile is not None:
        with open(appargs.conffile) as fconf:
            dconf = json.loads(fconf.read())
    if appargs.confstr is not None:
        dconf = json.loads(appargs.confstr)
    conf = Configuration()
    conf.update(dconf)
    conf['expr.city.id'] = CITY[conf['expr.city.name']]
    experiment(conf)
