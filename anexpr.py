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


import os
import re
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
source @0/opt_py27/opt_envs && mkdir -p /tmp/wl-tud && \
cd @1/twhere && python twhere/runner2.py -s '%(jstr)s' && \
hadoop fs -put %(output)s %(outdir)s && \
rm -rf %(output)s\
"""

SERVER_CONFSTR = """\
(mkdir -p %(outdir)s && python twhere/runner2.py -s \
'%(jstr)s') &\
"""

ILLEGAL_CHARS = re.compile(r'[\[\] \(\)\'"]')
UNICODE_LEADER = re.compile(r"u'")


def dict2str(adict):
    """ convert a dict into a string
    """
    s = '_'.join([('%s%s' % (k, v))
                  for k, v in adict.iteritems()
                  if k != 'expr.city.name'])
    s = UNICODE_LEADER.sub('', s)
    s = ILLEGAL_CHARS.sub('_', s)
    return s


def print_template(aconf, args):
    """ print the final script
    """
    print args.template % {'jstr': json.dumps(aconf),
                           'output': aconf['expr.output'],
                           'outdir': args.hadoop}


def print_withfolds(aconf, args):
    """ loop over folds
    """
    folds = DEFAULT_CONFIG['expr.folds']
    if 'expr.folds' in aconf:
        folds = aconf['expr.folds']
    outdir = 'test' \
        if args.outdir is None else args.outdir
    for f in range(folds):
        aconf['expr.fold_id'] = f
        aconf['expr.output'] = ''.join(
            [os.path.join(outdir, aconf['expr.city.name']),
             '_',
             args.exprname, '_',
             str(f),
             '.res'])
        print_template(aconf, args)


def expand_script(conf, args):
    """ Parsing the multiple parameters in the conf
    """
    param_names = [k for k, v in conf.iteritems() if isinstance(v, list)]
    for param_set in product(*[conf[k] for k in param_names]):
        aconf = dict(conf)
        aconf.update(dict([(k, v) for k, v in zip(param_names, param_set)]))
        args.exprname = args.name[0] + '_' + dict2str(aconf)
        if args.withfolds:
            print_withfolds(aconf, args)
        else:
            print_template(aconf, args)


def parse_parameter():
    """ Parse the argument
    """
    parser = argparse.ArgumentParser(
        description='Scripting experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Default Configuration:\n' + pformat(DEFAULT_CONFIG))
    parser.add_argument('-i',
                        dest='confstr',
                        action='store',
                        metavar='JSON',
                        default=None,
                        help='Running with the delta configuration '
                        'from the json string')
    parser.add_argument(dest='name',
                        action='store',
                        default='expr',
                        metavar='NAME',
                        nargs=1,
                        help='The name for the experiment')
    parser.add_argument('-e',
                        dest='expand',
                        action='store_true',
                        default=False,
                        help='Treat each delta object as a set of '
                        'parameters')
    arg_group = parser.add_mutually_exclusive_group()
    arg_group.add_argument('-b',
                           dest='hadoop',
                           action='store',
                           default=None,
                           metavar='HDFSDIR',
                           help='Generating sh script for BashWorkers '
                           'on Hadoop and store the res in HDFS')
    arg_group.add_argument('-s',
                           dest='server',
                           action='store_true',
                           default=False,
                           help='Generating sh script for multiprocess '
                           'pooling')
    parser.add_argument('-F',
                        dest='withfolds',
                        action='store_false',
                        default=True,
                        help='Generating scipts with different folds '
                        'for each concrete conf')
    parser.add_argument('-o',
                        dest='outdir',
                        default='test',
                        help='The folder to which the res files go')
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
    if args.hadoop:
        args.template = HADOOP_OPT_PY27_CONFSTR
    elif args.server:
        args.template = SERVER_CONFSTR
    else:
        args.template = '%(jstr)s'
    if args.expand is True:
        expand_script(conf, args)
    else:
        print_template(conf, args)


if __name__ == '__main__':
    utility()
