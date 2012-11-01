#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""File: trail.py
Description:
    Generating Trails
History: 0.1.0 The first version.
"""
__version__ = '0.1.0'
__author__ = 'SpaceLis'

import numpy as NP
import math
from datetime import datetime
import itertools
from model.colfilter import kernel_smooth


TRAILSECONDS = 24 * 3600


def TrailGen(seq, key=lambda x: x, diff=None):
    """ A generating function for trails
    """
    k = None
    trail = None
    u = None
    for s in seq:
        if k == key(s):
            if diff and u == diff(s):
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


class TimeParser(object):
    """ Parse time in the string/datetime object into seconds from midnight (from start)
    """
    def __init__(self, start=0, isstr=False):
        super(TimeParser, self).__init__()
        if isinstance(start, int):
            self.start = start
        elif isinstance(start, str):
            self.start = TimeParser.timedelta(start)
        else:
            raise ValueError('Malformed value for parameter: start')
        if isstr:
            self.parse = self.str2seconds
        else:
            self.parse = self.datetime2seconds

    def str2seconds(self, timestr):
        """ parse the timestr into seconds
        """
        t = datetime.strptime(timestr, '%Y-%m-%d %H:%M:%S')
        return (3600 * t.hour + 60 * t.minute + t.second - self.start) % TRAILSECONDS

    def datetime2seconds(self, tick):
        """ parse datetime object into seconds
        """
        return (3600 * tick.hour + 60 * tick.minute + tick.second - self.start) % TRAILSECONDS

    def timedelta(t):
        """ Convert string format of delta time into seconds
        """
        if not isinstance(t, str):
            raise ValueError('Expected str() but: ' + str(type(t)))
        if t[-1] == 'm':
            return int(t[:-1]) * 60
        elif t[-1] == 'h':
            return int(t[:-1]) * 3600

class Vectorizor(object):
    """ Make trail into vectors
    """
    def __init__(self, namespace, veclen, timeparser=TimeParser()):
        super(Vectorizor, self).__init__()
        self.namespace, self.veclen = namespace, veclen
        self.unit = float(TRAILSECONDS) / veclen
        self.timeparser = timeparser

    def process(trail):
        raise NotImplemented


class BinaryVectorizor(Vectorizor):
    """ Vectorizing trail as a 0-1 string each element of which indicates the presence of user at a location (type)
    """
    def __init__(self, namespace, veclen, timeparser=TimeParser(), isaccum=False):
        super(BinaryVectorizor, self).__init__(namespace, veclen, timeparser)
        if isaccum:
            self.process = self.process_accum
        else:
            self.process = self.process_binary
        self.timeparser = timeparser

    def process_binary(self, trail):
        """ This will only set the time slot to be one at the check-in location
        """
        vec = NP.zeros((len(self.namespace), self.veclen), dtype=NP.float32)
        for c in trail:
            poi_id = self.namespace.index(c['poi'])
            tickslot = math.trunc(self.timeparser.parse(c['tick']) / self.unit)
            vec[poi_id, tickslot] = 1

        return vec

    def process_accum(self, trail):
        """ This will add more weight if one time slot gets more check-in
        """
        vec = NP.zeros((len(self.namespace), self.veclen), dtype=NP.float32)
        for c in trail:
            poi_id = self.namespace.index(c['poi'])
            tickslot = math.trunc(self.timeparser.parse(c['tick']) / self.unit)
            vec[poi_id, tickslot] += 1
        return vec


class KernelVectorizor(Vectorizor):
    """ Using Gaussian shaped functions to model the likelihood of staying
    """
    def __init__(self, namespace, veclen, sigma=3600, timeparser=TimeParser(), bounded=False, isaccum=False, kernel='gaussian'):
        super(KernelVectorizor, self).__init__(namespace, veclen, timeparser)
        self.bounded = bounded
        self.kernel = kernel
        self.sigma = sigma
        if isaccum:
            self.aggr = NP.add
        else:
            self.aggr = NP.fmax

    def process(self, trail):
        """ accumulating gaussian shaped function
        """
        vec = NP.zeros((len(self.namespace), self.veclen), dtype=NP.float32)
        for poi, checkins in itertools.groupby(sorted(trail, key=lambda x: x['poi']), key=lambda x: x['poi']):
            idx = self.namespace.index(poi)
            vec[idx][:] = kernel_smooth([self.timeparser.parse(c['tick']) for c in checkins], self.sigma, self.veclen, aggr=self.aggr, kernel=self.kernel)
        vec[vec == 0] = 1.
        unity = NP.sum(vec, axis=0)
        NP.divide(vec, unity, vec)
        return vec


# --------------------
# TESTS
#
def testTrailGen():
    """ Test TrailGen
    """
    a = [[1, 2], [1, 4], [2, 3], [2, 3], [2, 7], [3, 7], [3, 7], [3, 3]]
    for t in TrailGen(a, lambda x: x[0]):
        print t


def testBinaryVectorizor():
    """
    """
    from datetime import datetime
    bvz = BinaryVectorizor(['a', 'b', 'c'], 24, isaccum=False)
    data = [
        "a 2010-12-13 04:07:12",
        "b 2010-12-13 04:20:39",
        "c 2010-12-13 06:09:51",
    ]
    tr = [{"poi": x, "tick": datetime.strptime(y, '%Y-%m-%d %H:%M:%S')} for x, y in [s.split(' ', 1) for s in data]]
    print bvz.process(tr)


def testKernelVectorizor():
    """
    """
    from datetime import datetime
    kvz = KernelVectorizor(['a', 'b', 'c'], 24)
    data = [
        "a 2010-12-13 04:07:12",
        "b 2010-12-13 04:20:39",
        "c 2010-12-13 06:09:51",
    ]
    tr = [{"poi": x, "tick": datetime.strptime(y, '%Y-%m-%d %H:%M:%S')} for x, y in [s.split(' ', 1) for s in data]]
    print kvz.process(tr)


if __name__ == '__main__':
    names = dict(globals())
    testfuncnames = [name for name in names if name.startswith('test')]
    for funcname in testfuncnames:
        globals()[funcname]()
