#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Evaluate the significance via wilcoxon.

File: evalsig.py
Author: SpaceLis
Email: Wen.Li@tudelft.nl
GitHub: http://github.com/spacelis
Description:


"""

import sys
import os
from collections import defaultdict
from mlmodels.evaluate.stats_R import wilcox
import re
WHITESPACE = re.compile(r'\s+')
MODELNAME = re.compile(r'\.model(.*)\.res')

def get_scope(fname):
    """ Return the scope of the res file so that results on the same dataset can be paired. """
    city = fname[:2]
    if 'targetbase' in fname:
        return city, 'base'
    elif 'targetcategory' in fname:
        return city, 'category'
    else:
        raise ValueError('%s do not have a scope' % fname)


def read_score(resfile):
    """ Read the scores from the res file, that is the res file is from the same city and the same level of location categories """
    with open(resfile) as fin:
        return [1 / float(WHITESPACE.split(l)[0]) for l in fin if not l.startswith('#')]


def modelname(s):
    """ return the model name"""
    try:
        return MODELNAME.search(s).group(1)
    except:
        return 'CF-K'

def printnotsig(res1, res2, city, level, threshold=0.05):
    """ Print x and y if they are not significant different, p is greater than threshold.

    :res1: the filepath of the .res file
    :res2: the filepath of another .res file
    :returns: None

    """

    x = read_score(res1)
    y = read_score(res2)
    if len(x) != len(y):
        m1 = modelname(res1)
        m2 = modelname(res2)
        print '%6s %2s %10s %30s %30s' % ('NA', city, level, m1, m2)
        return
    _, p = wilcox(x, y, paired=True)
    if p > threshold:
        m1 = modelname(res1)
        m2 = modelname(res2)
        print '%.4f %2s %10s %30s %30s' % (p, city, level, m1, m2)


def console(dirs):
    """ Gather the res files in different dirs and test the significance in each scope. """
    resfiles = [(get_scope(f), os.path.join(d, f)) for d in dirs for f in os.listdir(d) if f.endswith('.res')]
    scopes = defaultdict(list)
    for (city, level), f in resfiles:
        scopes[(city, level)].append(f)
    for (city, level), fs in scopes.items():
        for i in range(len(fs)):
            for j in range(i + 1, len(fs)):
                    printnotsig(fs[i], fs[j], city, level)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage: evalsig.py dir1 dir2 dir3 ...'
    console(sys.argv[1:])
