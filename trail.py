#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""File: trail.py
Description:
    Generating Trails
History: 0.1.0 The first version.
"""
__version__ = '0.1.0'
__author__ = 'SpaceLis'

import logging
logging.basicConfig(format='%(asctime)s %(name)s [%(levelname)s] %(message)s', level=logging.INFO)
LOGGER = logging.getLogger(__name__)

import numpy as NP
import math
from datetime import datetime
import itertools
from model.colfilter import kernel_smooth
from model.colfilter import KERNELS


TRAILSECONDS = 24 * 3600
EPSILON = 1e-20


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


def TrailSet(seq, minlen=1):
    """ Generating a trail set by truncating trail in to shorter lengths
    """
    for l in range(minlen, len(seq) + 1):
        yield seq[:l]


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


# -------------------------------------------------
# Vetorizors
#
class Vectorizor(object):
    """ Make trail into vectors
    """
    def __init__(self, namespace, veclen=100, timeparser=TimeParser()):
        super(Vectorizor, self).__init__()
        self.namespace, self.veclen = namespace, veclen
        self.unit = float(TRAILSECONDS) / veclen
        self.timeparser = timeparser

    def process(self, trail):
        raise NotImplemented

    def get_timeslot(self, t):
        raise NotImplemented


class BinaryVectorizor(Vectorizor):
    """ Vectorizing trail as a 0-1 string each element of which indicates the presence of user at a location (type)
    """
    def __init__(self, namespace, veclen=100, timeparser=TimeParser(), isaccum=False):
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
    def __init__(self, namespace, veclen=100, interval=(0., 24 * 3600.), kernel='gaussian', params=(3600,), isaccum=False, timeparser=TimeParser(), normalized=False):
        super(KernelVectorizor, self).__init__(namespace, veclen, timeparser)
        LOGGER.info('CONFIG: namespace=%d, veclen=%d, interval=%s, kernel=%s, params=%s, isaccum=%s, timeparser=%s, normalized=%s' % (len(namespace), veclen, str(interval), kernel, str(params), str(isaccum), str(timeparser), str(normalized)))
        self.veclen = veclen
        self.normalized = normalized
        self.kernel = kernel
        self.params = params
        self.axis = NP.linspace(*interval, num=veclen, endpoint=False)
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
            vec[idx][:] = kernel_smooth(self.axis, [self.timeparser.parse(c['tick']) for c in checkins], self.params, aggr=self.aggr, kernel=KERNELS[self.kernel])
        NP.add(vec, EPSILON, vec)
        if self.normalized:
            unity = NP.sum(vec, axis=0)
            NP.divide(vec, unity, vec)
        return vec

    def get_timeslot(self, t):
        """ Return the timeslot
        """
        return self.axis.searchsorted(self.timeparser.parse(t)) - 1


# -------------------------------------------------
# TESTS
#
def testTrailGen():
    """ Test TrailGen
    """
    a = [[1, 2], [1, 4], [2, 3], [2, 3], [2, 7], [3, 7], [3, 7], [3, 3]]
    for t in TrailGen(a, lambda x: x[0]):
        print t


def testTrailSet():
    """
    """
    a = [1, 2, 3, 4]
    for x in TrailSet(a):
        print x


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
    from matplotlib import pyplot as PLT
    from numpy.testing import assert_equal
    kvz = KernelVectorizor(['a', 'b', 'c'], 24)
    data = [
        "a 2010-12-13 04:07:12",
        "b 2010-12-13 04:20:39",
        "c 2010-12-13 06:09:51",
    ]
    tr = [{"poi": x, "tick": datetime.strptime(y, '%Y-%m-%d %H:%M:%S')} for x, y in [s.split(' ', 1) for s in data]]
    v = kvz.process(tr)
    PLT.plot(range(24), v[0])
    PLT.plot(range(24), v[1])
    PLT.plot(range(24), v[2])
    PLT.show()
    assert_equal(kvz.get_timeslot(datetime.strptime('2010-12-13 23:17:23', '%Y-%m-%d %H:%M:%S')), 23)


if __name__ == '__main__':
    names = dict(globals())
    testfuncnames = [name for name in names if name.startswith('test')]
    for funcname in testfuncnames:
        globals()[funcname]()
