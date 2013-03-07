#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: evalres.py
Author: SpaceLis
Changes:
    0.0.1 The first version
Description:
    Evaluate the results in directory by MRR
"""
__version__ = '0.0.1'


import os
from mlmodels.evaluate.mrr import mrr
import numpy as NP
from texttable import Texttable


PREFICES = ['NY', 'CH', 'LA', 'SF']


def eval_dir(path):
    """ Evaluate the results in the dir by MRR

        Argument:
            dirname -- the path to the diretory containing evaluation results
    """
    files = sorted(os.listdir(path))
    names = sorted(set([n.rsplit('.', 1)[0][3:] for n in files if n.endswith('.res')]))
    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.set_cols_dtype(['t', 'f', 'f', 'f', 'f'])
    table.set_cols_align(["l", "r", "r", "r", "r"])
    table.set_precision(4)
    table.add_row(['', ] + PREFICES)
    for n in names:
        scores = list()
        for prefix in PREFICES:
            try:
                eva = NP.array([int(v) for v in open(os.path.join(path, '%s_%s.res' % (prefix, n)))], dtype=NP.float64)
                scores.append(mrr(eva))
            except:
                scores.append('N/A')
        table.add_row([n, ] + scores)
    print table.draw()


if __name__ == '__main__':
    import sys
    eval_dir(sys.argv[1])
