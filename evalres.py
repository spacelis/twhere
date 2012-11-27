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
import site
site.addsitedir('../model')
from evaluate.mrr import mrr
import numpy as NP


PREFICES = ['NY', 'CH', 'LA', 'SF']


def eval_dir(path):
    """ Evaluate the results in the dir by MRR

        Argument:
            dirname -- the path to the diretory containing evaluation results
    """
    files = sorted(os.listdir(path))
    names = [n.split('.')[0][3:] for n in files if n.startswith('NY')]
    print '  ', '   \t'.join(names)
    for prefix in PREFICES:
        scores = list()
        for n in names:
            eva = NP.array([int(v) for v in open(os.path.join(path, '%s_%s.res' % (prefix, n)))], dtype=NP.float64)
            scores.append(mrr(eva))
        print prefix, '\t'.join([('%.4f' % v) for v in scores])


if __name__ == '__main__':
    import sys
    eval_dir(sys.argv[1])
