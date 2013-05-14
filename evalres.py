#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: evalres.py
Author: SpaceLis
Changes:
    0.0.2 Better table output
    0.0.1 The first version
Description:
    Evaluate the results in directory by MRR
"""
__version__ = '0.0.2'


import os
import re
import argparse
from mlmodels.evaluate.mrr import mrr
from mlmodels.evaluate.precision import precision
import numpy as NP
from texttable import Texttable


PREFICES = ['NY', 'CH', 'LA', 'SF']

prec1 = lambda x: precision(x, 1)

NUMBER = re.compile(r'[-]?\d+[.]?\d*')


def eval_dir(path, markdown=False, dprefix=False,
             evalmethod=None, numlabel=False):
    """ Evaluate the results in the dir by MRR

        Argument:
            dirname -- the path to the diretory containing evaluation results
    """
    if evalmethod is None:
        ef = mrr
    else:
        ef = globals()[evalmethod]
    files = sorted(os.listdir(path))
    names = sorted(set([n.rsplit('.', 1)[0][3:] for n in files
                        if n.endswith('.res')]),
                   key=lambda item: (len(item), item))
    if dprefix:
        prefices = sorted(set([n[:2] for n in files]))
    else:
        prefices = PREFICES
    table = Texttable()
    if markdown:
        table.set_asmarkdown()
    else:
        table.set_deco(0)
    table.set_cols_dtype(['t'] + ['f'] * len(prefices))
    table.set_cols_align(['l'] + ['r'] * len(prefices))
    table.set_precision(4)
    table.add_rows([['', ] + prefices])
    for n in names:
        scores = list()
        for prefix in prefices:
            try:
                eva = NP.array([int(v) for v in open(
                    os.path.join(path, '%s_%s.res' % (prefix, n)))],
                    dtype=NP.float64)
                scores.append(ef(eva))
            except IOError:
                scores.append('N/A')
        if numlabel:
            n = ' '.join([m.group() for m in NUMBER.finditer(n)])
        table.add_row([n, ] + scores)
    print table.draw()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluating experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-m',
                        dest='markdown',
                        action='store_true',
                        default=False,
                        help='Enable markdown style')
    parser.add_argument('-p',
                        dest='dprefix',
                        action='store_true',
                        default=False,
                        help='Dynamic changing prefix according '
                        'to the content in resdir')
    parser.add_argument(dest='resdir',
                        action='store',
                        metavar='NAME',
                        nargs=1,
                        help='The name for the experiment')
    parser.add_argument('-e',
                        dest='evalmethod',
                        action='store',
                        default=None,
                        help='the evaluation method')
    parser.add_argument('-s',
                        dest='numlabel',
                        action='store_true',
                        default=False,
                        help='Print shorter labels for parameters')
    args = parser.parse_args()
    eval_dir(args.resdir[0], args.markdown, args.dprefix, args.evalmethod, args.numlabel)
