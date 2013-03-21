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
import yaml
import json
import logging
import logging.config
from twhere.exprmodels import experiment
from twhere.config import Configuration

CITY = dict(zip(['NY', 'CH', 'LA', 'SF'],
           ['27485069891a7938', '1d9a5370a355ab0c', '3b77caf94bfc81fe', '5a110d312052166f']))

LOGGING_CONF = """
version: 1
formatters:
    simple:
        format: "%(asctime)s %(name)s [%(levelname)s] %(message)s"

handlers:
    console:
        class: logging.StreamHandler
        level: INFO
        formatter: simple
        stream: ext://sys.stderr

root:
    level: INFO
    handlers: [console, info_file_handler]
"""


def setup_logging(logstring):
    """ Setup logging
    """
    logconf = yaml.load(logstring)
    logging.config.dictConfig(logconf)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print >> sys.stderr, 'No folder provided for results'
        print >> sys.stderr, 'Usage: runtest.py <resdir> <conf>'
        sys.exit(-1)
    resdir = sys.argv[1]
    conffile = sys.argv[2]
    if not os.path.isdir(resdir):
        os.mkdir(resdir)
    setup_logging(LOGGING_CONF)
    LOGGER = logging.getLogger(__name__)
    LOGGER.debug('DEBUG is enabled')
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (2.5 * 1024 * 1024 * 1024L, -1L))
    except ValueError as err:
        LOGGER.warn('Failed set resource limits. Because {0}'.format(err.message))

    with open(conffile) as fconf:
        dconf = json.loads(fconf.read())
    conf = Configuration()
    conf.update(dconf)
    conf['expr.city.id'] = CITY[conf['expr.city.name']]
    experiment(conf)
