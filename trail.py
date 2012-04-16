#!/bin/python
# -*- coding: utf-8 -*-
"""File: trail.py
Description:
    Generating Trails
History: 0.1.0 The first version.
"""
__version__ = '0.1.0'
__author__ = 'SpaceLis'


def TrailGen(seq, key=lambda x: x, diff=None):
    """ A generating function for trails
    """
    k = None
    trail = None
    u = None
    for s in seq:
        if k == key(s):
            if diff and u==diff(s):
                continue
            trail.append(s)
        elif k:
            yield trail
            trail = list()
            trail.append(s)
            k = key(s)
        else:
            trail = list()
            trail.append(s)
            k = key(s)
        if diff:
            u = diff(s)
    yield trail

def test():
    """ Test TrailGen
    """
    a = [[1,2], [1,4], [2,3], [2,3], [2,7],[3,7],[3,7],[3,3]]
    for t in TrailGen(a, lambda x:x[0]):
        print t

if __name__ == '__main__':
    import sys
    print sys.path
    print 'Test'
    test()


