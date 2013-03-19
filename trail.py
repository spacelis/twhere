#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""File: trail.py
Description:
    Generating Trails
History:
    0.2.0 long trail scheme
    0.1.1 sufficient tests for existing functions
    0.1.0 The first version.
"""
__version__ = '0.2.0'
__author__ = 'SpaceLis'

import logging
logging.basicConfig(format='%(asctime)s %(name)s [%(levelname)s] %(message)s', level=logging.INFO)
LOGGER = logging.getLogger(__name__)

import math
from datetime import datetime, timedelta
import itertools
from pprint import pformat
import numpy as NP
from numpy.lib.stride_tricks import as_strided


EPSILON = 1e-10

# ----------------------------------------------------------------------------------------------------
# KERNELS
#
def gaussian_pdf(axis, mu, params):
    """ Generate a sample of Gaussian distribution (mu, ss) in interval with veclen samples

        Arguments:
            axis -- a vector of x coordinates each element of which is a sample point
            mu -- a float indication the mean of Gaussian distribution
            sigma -- a float indicating the sigma of Gaussian distribution
    """
    (sigma,) = params
    s = (axis - mu) / sigma
    return NP.exp(-NP.square(s))


def uniform_pdf(axis, mu, params):  #  mu is a compromise to the interface pylint: disable-msg=W0613
    """ Generate a sample of uniform distribution

        Arguments:
            axis -- a vector of x coordinates each element of which is a sample point
            mu -- a float indication the mean of Gaussian distribution
            sigma -- a float indicating the sigma of Gaussian distribution
    """
    return NP.ones_like(axis)


def kernel_smooth(axis, mus, params, aggr=NP.add, kernel=gaussian_pdf):
    """ Cumulating the Gaussian distribution

        Arguments:
            mus -- a set of mus
            params -- the parameters for kernels, e.g., sigma for Gaussian
            axis -- a vector of x coordinates each element of which is a sample point
            aggr -- the aggragation method to be used for the set of Gaussian distribution. Default=NP.add
            kernel -- name of kernel distribution to use for smoothing
    """
    vec = NP.zeros_like(axis)
    for x in mus:
        aggr(kernel(axis, x, params), vec, vec)
    return vec


def checkin_trails(seq, key=lambda x: x['trail_id']):
    """ A generating function for trails

        Arguments:
            seq -- a sequence of check-ins interpreted as a dict() containing at least a key of 'trail_id'
    """
    return [list(cks) for _, cks in itertools.groupby(seq, key=key)]


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
    for _, trls in itertools.groupby(sorted(list(trails), key=key), key=key):
        trls = list(trls)
        for _ in range(maximum):
            yield trls[NP.random.randint(len(trls))]


# -------------------------------------------------
# Trail segmentation
#

class TrailVectorSegmentSet(object):
    """ Cutting vectorized trails in to segments along time dimension according to given reference point.
    """
    def __init__(self, data, seglen=100):
        """ Create a view as a vector of trail segments.
        Each element is a matrix of time [(of the given seglen) x user x poi].
        For example, as_segments([A, B, C, D, E], 3) == [[A, B, C], [B, C, D], [C, D, E])

            Arguments:
                data -- a 3D NP.array() [time, user, poi]
                seglen -- the length of segments in timeslots

            Return:
                a 5d array [offset, seg, time, user, poi]
        """
        assert len(data.shape) == 3, 'Trails must be stored as [time] * [user] * [poi]'
        super(TrailVectorSegmentSet, self).__init__()
        self.length, self.user_num, self.poi_num = data.shape
        self.seglen = seglen

        self.frame_num = self.length - self.seglen + 1  # Frames are sliding windows of segment length
        self.snapshot_bytesize = self.user_num * self.poi_num * data.itemsize
        self.segsets = as_strided(data,
                                  shape=(self.frame_num, seglen, self.user_num, self.poi_num),
                                  strides=(self.snapshot_bytesize, self.snapshot_bytesize, self.poi_num * data.itemsize, data.itemsize))

    def enumerate_segment_sets(self, refslot):
        """ Enumerating segment sets according to reference point
        """
        for i in NP.arange((refslot + 1) % self.seglen, refslot, self.seglen):  # including the refslot point as the last time point in a segment
            yield i, self.segsets[i]

    def enumerate_segments(self, refslot):
        """ Enumerating segments
        """
        for t, segset in self.enumerate_segment_sets(refslot):
            for u in range(self.user_num):
                yield u, t, segset


def as_vector_segments(data, ref, seglen):
    """ return a 4d matrix to represent segments,

        Arguments:
            data -- a 3d array [user, poi, time]
            ref -- the reference point of time in terms of timeslot (int)
            seglen -- the length of a segment

        Return:
            4d array [user, segment, poi, time]
    """
    user_num, poi_num, length = data.shape

    offset = ((ref + 1) % seglen)
    segment_num = (length - offset) / seglen  # Frames are non-overlapping sliding windows of segment length, because they are decided on-the-fly of prediction
    longtrail_bytesize = poi_num * length * data.itemsize
    return as_strided(data,
                      shape=(user_num, segment_num, 2, poi_num, seglen),
                      strides=(longtrail_bytesize,
                               seglen * data.itemsize, # shoundn't this be (seglen * poi_num * data.itemsize)
                               offset * data.itemsize,
                               length * data.itemsize,
                               data.itemsize))[:, :, 1, :, :]


# This method is introduced for sparse vectors as a segment starting at any
# time point is constructed by two neighboring vectors. Thus segments are not
# overlapping.
def as_segments(vec, seglen, ref=-1):
    """ return a 3d matrix to represent segments for one trail.

        Arguments:
            vec -- a 2d array [poi, time]
            seglen -- the length of a segment
            ref -- the reference point of time in terms of timeslot (int)

        Return:
            a 3d array [segment, poi, time]
    """
    poi_num, length = vec.shape
    offset = ((ref + 1) % seglen)
    segment_num = (length - offset) / seglen  # Frames are non-overlapping sliding windows of segment length
    return as_strided(vec,
                      shape=(2, segment_num, poi_num, seglen),
                      strides=(offset * vec.itemsize,
                               seglen * vec.itemsize,
                               length * vec.itemsize,
                               vec.itemsize))[1, :, :, :]


def as_doublesegments(vec, seglen, ref=-1):
    """ return a 3d matrix to represent double segments for one trail.

        Arguments:
            vec -- a 2d array [poi, time]
            seglen -- the length of a segment
            ref -- the reference point of time in terms of timeslot (int)

        Return:
            a 3d array [segment, poi, time]
    """
    poi_num, length = vec.shape
    offset = ((ref + 1) % seglen)
    segment_num = (length - offset) / seglen - 1  # Frames are non-overlapping sliding windows of segment length
    return as_strided(vec,
                      shape=(2, segment_num, poi_num, 2 * seglen),
                      strides=(offset * vec.itemsize,
                               seglen * vec.itemsize,
                               length * vec.itemsize,
                               vec.itemsize))[1, :, :, :]


def as_mask(data, ref, seglen, level=1):
    """ Create a histogram for prune calculations of useless vectors
    """
    user_num, length = data.shape

    offset = ((ref + 1) % seglen)
    segment_num = (length - offset) / seglen  # Frames are sliding windows of segment length
    longtrail_bytesize = length * data.itemsize
    segs = as_strided(data,
                      shape=(user_num, segment_num, 2, seglen),
                      strides=(longtrail_bytesize,
                               seglen * data.itemsize,
                               offset * data.itemsize,
                               data.itemsize))[:, :, 1, :]
    return (NP.sum(segs, axis=2) > level)


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
    def __init__(self,                       #  all argumnets are required pylint: disable-msg=R0913
                 namespace,
                 unit=timedelta(seconds=24 * 36),
                 epoch=datetime(2010, 6, 1),
                 eschatos=datetime(2011, 6, 1),
                 timeparser=None):

        super(Vectorizor, self).__init__()
        self.namespace = namespace
        self.unit = unit if isinstance(unit, timedelta) else str2timedelta(unit)
        self.epoch = epoch
        self.eschatos = eschatos

        self.unit = unit.total_seconds()
        self.veclen = (self.eschatos - self.epoch).total_seconds() / self.unit
        self.timeparser = timeparser if timeparser is not None else lambda x: x

    def process(self, trail, target=None):
        """ Processing trails to vectors
        """
        raise NotImplementedError

    def get_timeslot(self, tick):
        """ return current timeslot the tick belongs to
        """
        return math.trunc(self.get_seconds(tick) / self.unit)

    def get_seconds(self, tick):
        """ get the total seconds from epoch to tick
        """
        return (self.timeparser(tick) - self.epoch).total_seconds()

    def __str__(self):
        attrs = [(k, repr(getattr(self, k))) for k in dir(self)]
        attrs = [(k, v) for k, v in attrs if not (k.startswith('_') or hasattr(getattr(self, k), 'im_self'))]
        attrs = [(k, v) if len(v) < 50 else (k, v[:47] + '...') for k, v in attrs]
        return pformat(dict(attrs))


class BinaryVectorizor(Vectorizor):                 # process method is linked in __init__ pylint: disable-msg=W0223
    """ Vectorizing trail as a 0-1 string each element of which indicates the presence of user at a location (type)
    """
    def __init__(self, namespace, accumulated=True, **kargs):

        super(BinaryVectorizor, self).__init__(namespace, **kargs)
        if accumulated:
            self.process = self.process_accum
        else:
            self.process = self.process_binary
        LOGGER.info('CONFIG [{0}]: {1}'.format(type(self), str(self)))

    @classmethod
    def from_config(cls, conf):
        """ Constructing from conf
        """
        return cls(conf['data.namespace'],
                   accumulated=conf['vec.kernel.accumulated'],
                   unit=conf['vec.unit'],
                   epoch=conf['vec.epoch'],
                   eschatos=conf['vec.eschatos'],
                   timeparser=conf['vec.timeparser'])

    def process_binary(self, trail, target=None):
        """ This will only set the time slot to be one at the check-in location
        """
        if target is not None:
            vec = target
        else:
            vec = NP.zeros((len(self.namespace), self.veclen), dtype=NP.float32)
        for c in trail:
            poi_id = self.namespace.index(c['poi'])
            tickslot = self.get_timeslot(c['tick'])
            vec[poi_id, tickslot] = 1
        return vec

    def process_accum(self, trail, target=None):
        """ This will add more weight if one time slot gets more check-in
        """
        if target is not None:
            vec = target
        else:
            vec = NP.zeros((len(self.namespace), self.veclen), dtype=NP.float32)
        for c in trail:
            poi_id = self.namespace.index(c['poi'])
            tickslot = self.get_timeslot(c['tick'])
            vec[poi_id, tickslot] += 1
        return vec


class KernelVectorizor(Vectorizor):
    """ Using Gaussian shaped functions to model the likelihood of staying
    """
    def __init__(self,                        #  all argumnets are required pylint: disable-msg=R0913
                 namespace,
                 kernel='gaussian_pdf',
                 params=(3600,),
                 accumulated=False,
                 normalized=False,
                 **kargs):
        super(KernelVectorizor, self).__init__(namespace, **kargs)
        self.kernel = globals()[kernel]
        self.params = params
        self.normalized = normalized
        self.interval = (0, (self.eschatos - self.epoch).total_seconds())
        self.axis = NP.linspace(*self.interval, num=self.veclen, endpoint=False)
        if accumulated:
            self.aggr = NP.add
        else:
            self.aggr = NP.fmax
        LOGGER.info('CONFIG [{0}]: {1}'.format(type(self), str(self)))

    @classmethod
    def from_config(cls, conf):
        """ Constructing from configurations
        """
        return cls(conf['data.namespace'],
                   kernel=conf['vec.kernel'],
                   params=conf['vec.kernel.params'],
                   accumulated=conf['vec.kernel.accumulated'],
                   normalized=conf['vec.kernel.normalized'],
                   unit=conf['vec.unit'],
                   epoch=conf['vec.epoch'],
                   eschatos=conf['vec.eschatos'],
                   timeparser=conf['vec.timeparser'])

    def process(self, trail, target=None):
        """ accumulating gaussian shaped function
        """
        if target is not None:
            vec = target
        else:
            vec = NP.zeros((len(self.namespace), self.veclen), dtype=NP.float32)
        for poi, checkins in itertools.groupby(sorted(trail, key=lambda x: x['poi']), key=lambda x: x['poi']):
            idx = self.namespace.index(poi)
            vec[idx, :] = kernel_smooth(self.axis,
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
        return self.axis.searchsorted(self.get_seconds(t)) - 1  # axis is an array of sample points pylint: disable-msg=E1103
