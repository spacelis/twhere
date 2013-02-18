#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""File: trail.py
Description:
    Generating Trails
History:
    0.1.1 sufficient tests for existing functions
    0.1.0 The first version.
"""
__version__ = '0.1.1'
__author__ = 'SpaceLis'

import logging
logging.basicConfig(format='%(asctime)s %(name)s [%(levelname)s] %(message)s', level=logging.INFO)
LOGGER = logging.getLogger(__name__)

import numpy as NP
from numpy.lib.stride_tricks import as_strided
import math
from datetime import datetime, timedelta
import itertools
from model.colfilter import kernel_smooth
from model.colfilter import gaussian_pdf


EPSILON = 1e-10


def iter_checkin_trails(seq, key=lambda x: x['trail_id']):
    """ A generating function for trails

        Arguments:
            seq -- a sequence of check-ins interpreted as a dict() containing at least a key of 'trail_id'
    """
    return itertools.groupby(seq, key=key)


# -------------------------------------------------
# Trail filters
#

def length_limits_filter(trails, minlen=2, maxlen=10000):
    """ Filter out shorter trails
    """
    return itertools.ifilter(lambda x: minlen < x < maxlen, trails)


def diff_last_filter(trails, key=lambda x: x['pid']):
    """ Filter out trails with last two key different
    """
    return itertools.ifilter(lambda x: key(x[-1]) != key(x[-2]), trails)


def diff_all_filter(trails, key=lambda x: x['pid']):
    """ Filter out trails with last key appeared before
    """
    return itertools.ifilter(lambda x: key(x[-1]) not in set([key(c) for c in x]), trails)


def uniformed_last_filter(trails, key=lambda x: x['poi'], maximum=1):
    """ Filter trails so that each 'poi' has exactly one trail ends with it
    """
    for poi, trls in itertools.groupby(sorted(list(trails), key=key), key=key):
        trls = list(trls)
        for _ in range(maximum):
            yield trls[NP.random.randint(len(trls))]


# -------------------------------------------------
# Trail segmentation
#

class TrailVectorSegmentSet(object):
    """ Cutting vectorized trails in to segments along time dimension according to given reference point.
    """
    def __init__(self, data, ref, seglen=100):
        """ Create a view as a vector of trail segments.
        Each element is a matrix of time [(of the given seglen) x user x poi].
        For example, as_segments([A, B, C, D, E], 3) == [[A, B, C], [B, C, D], [C, D, E])

            Arguments:
                data -- a 3D NP.array() time x user x poi
                ref -- a timeslot used as reference
                seglen -- the length of segments in timeslots
        """
        assert len(data.shape) == 3, 'Trails must be stored as [time] * [user] * [poi]'
        super(TrailVectorSegmentSet, self).__init__()
        self.length, self.user_num, self.poi_num = data.shape
        self.seglen = seglen
        self.ref = ref

        self.frame_num = self.length - self.seglen + 1  # Frames are sliding windows of segment length
        self.snapshot_bytesize = self.user_num * self.poi_num * data.itemsize
        self.segsets = as_strided(data,
                                  shape=(self.frame_num, seglen, self.user_num, self.poi_num),
                                  strides=(self.snapshot_bytesize, self.snapshot_bytesize, self.poi_num * data.itemsize, data.itemsize))

    def enumerate_segment_sets(self):
        """ Enumerating segment sets according to reference point
        """
        for i in NP.arange((self.ref + 1) % self.seglen, self.ref, self.seglen):  # including the ref point as the last time point in a segment
            yield i, self.segsets[i]

    def enumerate_segments(self):
        """ Enumerating segments
        """
        for t, segset in self.enumerate_segment_sets():
            for u in range(self.user_num):
                yield u, t, segset


# -------------------------------------------------
# Vetorizors
#

def str2datetime(s):
    """ Parse a string into datetime
    """
    datetime.strptime(s, '%Y-%m-%d %H:%M:%S')


def str2timedelta(s):
    """ Convert string format of delta time into timedelta
    """
    if not isinstance(s, str):
        raise ValueError('Expected str() but: ' + str(type(s)))
    if s[-1] == 'm':
        return timedelta(minutes=int(s[:-1]))
    elif s[-1] == 'h':
        return timedelta(hours=int(s[:-1]))


class Vectorizor(object):
    """ Make trail into vectors
    """
    def __init__(self,
                 namespace,
                 unit=timedelta(seconds=24 * 36),
                 epoch=datetime(2010, 6, 1),
                 eschatos=datetime(2011, 6, 1),
                 timeparser=lambda x: x):

        super(Vectorizor, self).__init__()
        self.namespace = namespace
        self.unit = unit if isinstance(unit, timedelta) else str2timedelta(unit)
        self.epoch = epoch
        self.eschatos = eschatos
        self.timeparser = timeparser

        self.unit = unit.total_seconds()
        self.veclen = (self.eschatos - self.epoch).total_seconds() / self.unit
        self.timeparser = timeparser

    def process(self, trail, target=None):
        raise NotImplemented

    def get_timeslot(self, tick):
        return math.trunc(self.get_seconds(tick) / self.unit)

    def get_seconds(self, tick):
        return (self.timeparser(tick) - self.epoch).total_seconds()


class BinaryVectorizor(Vectorizor):
    """ Vectorizing trail as a 0-1 string each element of which indicates the presence of user at a location (type)
    """
    def __init__(self, namespace, isaccum=True, **kargs):

        super(BinaryVectorizor, self).__init__(namespace, **kargs)
        #LOGGER.info('CONFIG: namespace=%d, veclen=%d, interval=%s, kernel=%s, params=%s, isaccum=%s, timeparser=%s, normalized=%s' % (len(namespace), veclen, str(interval), kernel, str(params), str(isaccum), str(timeparser), str(normalized)))
        if isaccum:
            self.process = self.process_accum
        else:
            self.process = self.process_binary
        config = dict([(k, getattr(self, k)) for k in dir(self) if not (k.startswith('_') or hasattr(getattr(self, k), 'im_self'))])
        LOGGER.info('CONFIG [{0}]: {1}'.format(type(self), config))

    def process_binary(self, trail, target=None):
        """ This will only set the time slot to be one at the check-in location
        """
        if target is not None:
            vec = target
        else:
            vec = NP.zeros((self.veclen, len(self.namespace)), dtype=NP.float32)
        for c in trail:
            poi_id = self.namespace.index(c['poi'])
            tickslot = self.get_timeslot(c['tick'])
            vec[tickslot, poi_id] = 1
        return vec

    def process_accum(self, trail, target=None):
        """ This will add more weight if one time slot gets more check-in
        """
        if target is not None:
            vec = target
        else:
            vec = NP.zeros((self.veclen, len(self.namespace)), dtype=NP.float32)
        for c in trail:
            poi_id = self.namespace.index(c['poi'])
            tickslot = self.get_timeslot(c['tick'])
            vec[tickslot, poi_id] += 1
        return vec


class KernelVectorizor(Vectorizor):
    """ Using Gaussian shaped functions to model the likelihood of staying
    """
    def __init__(self, namespace, kernel=gaussian_pdf, params=(3600,), isaccum=False, normalized=False, **kargs):
        super(KernelVectorizor, self).__init__(namespace, **kargs)
        self.kernel = kernel
        self.params = params
        self.normalized = normalized
        self.interval = (0, (self.eschatos - self.epoch).total_seconds())
        self.axis = NP.linspace(*self.interval, num=self.veclen, endpoint=False)
        if isaccum:
            self.aggr = NP.add
        else:
            self.aggr = NP.fmax
        config = dict([(k, getattr(self, k)) for k in dir(self) if not (k.startswith('_') or hasattr(getattr(self, k), 'im_self'))])
        LOGGER.info('CONFIG [{0}]: {1}'.format(type(self), config))

    def process(self, trail, target=None):
        """ accumulating gaussian shaped function
        """
        if target is not None:
            vec = target
        else:
            vec = NP.zeros((self.veclen, len(self.namespace)), dtype=NP.float32)
        for poi, checkins in itertools.groupby(sorted(trail, key=lambda x: x['poi']), key=lambda x: x['poi']):
            idx = self.namespace.index(poi)
            vec[:, idx] = kernel_smooth(self.axis,
                                        [self.get_seconds(c['tick']) for c in checkins],
                                        self.params,
                                        aggr=self.aggr,
                                        kernel=self.kernel)
        NP.add(vec, EPSILON, vec)
        if self.normalized:
            unity = NP.sum(vec, axis=0)
            NP.divide(vec, unity, vec)
        return vec

    def get_timeslot(self, t):
        """ Return the timeslot
        """
        return self.axis.searchsorted(self.get_seconds(t)) - 1
