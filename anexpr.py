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
hadoop fs -put %(output)s %(outdir)s && \
rm %(output)s)\
"""

HADOOP_OPT_PY27_CONFSTR = """\
source @0/opt_py27/opt_envs && \
(cd @1/twhere && python twhere/runner2.py -s \
'%(jstr)s' && mkdir -p /tmp/wl-tud && \
hadoop fs -put /tmp/wl-tud/%(output)s %(outdir)s && \
rm /tmp/wl-tud/%(output)s\
"""

SERVER_CONFSTR = """\
(mkdir -p %(dir)s && python twhere/runner2.py -s \
'%(jstr)s') &\
"""

CITY = ['NY', 'CH', 'LA', 'SF']


def dict2str(adict):
    """ convert a dict into a string
    """
    return '_'.join([('%s_%s' % (k, v)) for k, v in adict.iteritems()])


def print_cityloop(conf, args):
    """ loop over city
    """
    exprname = args.exprname
    outdir = 'test' \
        if args.outdir is None else args.outdir
    aconf = dict(conf)
    for c in CITY:
        aconf['expr.city.name'] = c
        for f in range(10):
            aconf['expr.fold_id'] = f
            aconf['expr.output'] = ''.join([c,
                                           '_',
                                           exprname,
                                           '_',
                                           dict2str(conf),
                                           '_',
                                           str(f),
                                           '.res'])
            print args.template % {'jstr': json.dumps(aconf),
                                   'output': aconf['expr.output'],
                                   'outdir': outdir}


def print_script(conf, args):
    """ Parsing the multiple parameters in the conf
    """
    param_names = [k for k, v in conf.iteritems if isinstance(v, list)]
    for param_set in product(*[conf[param_names[k]] for k in param_names]):
        c = dict(conf)
        c.update(dict([(k, v) for k, v in zip(param_names, param_set)]))
        print_cityloop(conf, args)


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
    parser.add_argument('-h',
                        dest='hadoop',
                        action='store_true',
                        default=False,
                        help='Generating sh script for BashWorkers'
                        'on Hadoop')
    parser.add_argument('-o', '--output-dir',
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
    args.template = HADOOP_OPT_PY27_CONFSTR \
        if args.hadoop else SERVER_CONFSTR
    if args.expand is True:
        print_script(conf, args)
    else:
        print_cityloop(conf, args)


if __name__ == '__main__':
    utility()
