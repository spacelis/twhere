#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: anexpr.py
Author: SpaceLis
Changes:
    0.0.1 The first version
Description:
"""
__version__ = '0.0.1'


import json
import argparse
from itertools import product
from pprint import pformat

from twhere.config import DEFAULT_CONFIG

# The folloing string template is used for generating a bash script for running
# experiments on hadoop where @0 is a place holder for a pre-packaged python
# evironment while @1 will be replaced by the actual path to the folder of
# twhere. The pydo lib will use distributed cache for storing local copy of
# the python env and experiment codes that is why @0 and @1 are used here.
HADOOP_CONFSTR = """\
source @0/py27/bin/activate && \
(cd @1/twhere && python twhere/runner2.py -s \
'%(jstr)s' && \
hadoop fs -put %(tmpprefix)s%(output)s %(outdir)s && \
rm %(output)s)\
"""

HADOOP_OPT_PY27_CONFSTR = """\
source @0/opt_py27/opt_envs && \
(cd @1/twhere && python twhere/runner2.py -s \
'%(jstr)s' && \
hadoop fs -put %(tmpprefix)s%(output)s %(outdir)s && \
rm %(output)s)\
"""

SERVER_CONFSTR = """\
(mkdir -p %(dir)s && python twhere/runner2.py -s \
'%(jstr)s' 2> %(log)s) &\
"""

CITY = ['NY', 'CH', 'LA', 'SF']


def dict2str(adict):
    """ convert a dict into a string
    """
    return '_'.join([('%s_%s' % (k, v)) for k, v in adict.iteritems()])


def print_cityloop_hadoop(conf,
                          exprname,
                          tmpprefix='/tmp/wl-tudelft-',
                          outdir='test'):
    """ loop over city
    """
    aconf = dict(conf)
    for c in CITY:
        aconf['expr.city.name'] = c
        for f in range(10):
            aconf['expr.fold_id'] = f
            aconf['expr.output'] = ''.join([tmpprefix,
                                           c,
                                           '_',
                                           exprname,
                                           dict2str(conf),
                                           '_',
                                           str(f),
                                           '.res'])
            print HADOOP_OPT_PY27_CONFSTR % {'jstr': json.dumps(aconf),
                                             'tmpprefix': tmpprefix,
                                             'output': aconf['expr.output'],
                                             'outdir': outdir}


def print_cityloop_server(conf, exprname, outdir='test'):
    """ loop over city
    """
    aconf = dict(conf)
    for c in CITY:
        aconf['expr.city.name'] = c
        for f in range(10):
            aconf['expr.fold_id'] = f
            aconf['expr.output'] = ''.join([outdir,
                                           '/',
                                           c,
                                           '_',
                                           exprname,
                                           '_',
                                           str(f),
                                           '.res'])
            print SERVER_CONFSTR % {'jstr': json.dumps(aconf),
                                    'dir': outdir,
                                    'log': aconf['expr.output'] + '.log'}


def parse_parameter():
    """ Parse the argument
    """
    parser = argparse.ArgumentParser(
        description='Scripting experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Default Configuration:\n' + pformat(DEFAULT_CONFIG))
    parser.add_argument('-s',
                        dest='confstr',
                        action='store',
                        metavar='JSON',
                        default=None,
                        help='Running with the delta configuration '
                        'from the json string')
    parser.add_argument('-n',
                        dest='name',
                        required=True,
                        action='store',
                        default='expr',
                        metavar='NAME',
                        help='The name for the experiment')
    parser.add_argument('-e',
                        dest='expand',
                        action='store_true',
                        default=False,
                        help='Treat each delta object as a set of '
                        'parameters')
    subparsers = parser.add_subparsers(dest='cmd')
    parser_hadoop = subparsers.add_parser('hadoop')
    parser_server = subparsers.add_parser('server')
    parser_hadoop.add_argument('-t', '--tmpprefix',
                               dest='tmpprefix',
                               default='/tmp/wl-tudelft-',
                               help='The the prefix for accessing tmp '
                               'folder with name')
    parser_hadoop.add_argument('-o', '--output-dir',
                               dest='output',
                               default='test',
                               help='The folder on HDFS for gathering '
                               'the result files')

    parser_server.add_argument('-o', '--output-dir',
                               dest='output',
                               default='test',
                               help='The folder on HDFS for gathering '
                               'the result files')
    args = parser.parse_args()
    return args


def utility():
    """ Running this toolkit
    """
    args = parse_parameter()
    if args.confstr is None:
        conf = dict()
    else:
        conf = json.loads(args.confstr)

    if args.expand is True:
        valueset = list(conf.itervalues())
        for a in product(*valueset):
            aconf = dict(zip(conf.keys(), a))
            if args.cmd == 'hadoop':
                print_cityloop_hadoop(aconf,
                                      args.name,
                                      args.tmpprefix,
                                      args.output)
            elif args.cmd == 'server':
                print_cityloop_server(aconf, args.name)
    else:
        if args.cmd == 'hadoop':
            print_cityloop_hadoop(conf, args.name, args.tmpprefix, args.output)
        elif args.cmd == 'server':
            print_cityloop_server(conf, args.name)

if __name__ == '__main__':
    utility()
