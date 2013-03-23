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
rm %(tmpprefix)s%(output)s)\
"""

SERVER_CONFSTR = """\
(mkdir -p %(dir)s && python twhere/runner2.py -s \
'%(jstr)s' ) &
"""

CITY = ['NY', 'CH', 'LA', 'SF']


def print_cityloop_hadoop(conf,
                          exprname,
                          tmpprefix='/tmp/wl-tudelft-',
                          outdir='test'):
    """ loop over city
    """
    for c in CITY:
        conf['expr.city.name'] = c
        for f in range(10):
            conf['expr.fold_id'] = f
            conf['expr.output'] = ''.join([tmpprefix, c, '_',
                                           exprname, '_', str(f), '.res'])
            print HADOOP_CONFSTR % {'jstr': json.dumps(conf),
                                    'tmpprefix': tmpprefix,
                                    'output': conf['expr.output'],
                                    'outdir': outdir}


def print_cityloop_server(conf, exprname, outdir='test'):
    """ loop over city
    """
    for c in CITY:
        conf['expr.city.name'] = c
        for f in range(10):
            conf['expr.fold_id'] = f
            conf['expr.output'] = ''.join([outdir, '/', c, '_',
                                           exprname, '_', str(f)])
            print SERVER_CONFSTR % {'jstr': json.dumps(conf),
                                    'dir': outdir}


def parse_parameter():
    """ Parse the argument
    """
    parser = argparse.ArgumentParser(description='Scripting experiments')
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
    if args.cmd == 'hadoop':
        print_cityloop_hadoop(conf, args.name, args.tmpprefix, args.output)
    elif args.cmd == 'server':
        print_cityloop_server(conf, args.name)

if __name__ == '__main__':
    utility()
