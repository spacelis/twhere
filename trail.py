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
import math
from datetime import datetime
import itertools
from model.colfilter import kernel_smooth
from model.colfilter import gaussian_pdf


TRAILSECONDS = 24 * 3600
EPSILON = 1e-10


#FIXME diff parameter should be tested
def itertrails(seq, key=lambda x: x, diff=None):
    """ A generating function for trails

        Arguments:
            seq -- a trail
            key -- a key function used for segmenting a trail into several trails, working like itertools.groupby()
            diff -- a function making each generated trail has no consecutive checkins at the same place.
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


def iter_subtrails(seq, minlen=2, diffmode=None, diffkey=None):
    """ Generating a trail set by truncating trail in to shorter lengths

        Arguments:
            seq -- a trail
            minlen -- a minimum length of genereated trails
            diffmode -- None: all subtrails,
                        'last': subtrails with different last two according to diffkey
                        'all': subtrails with the last different from the rest
            diffkey -- a function of differeciating the last two elements,
                    making sure returned subtrails with key(subtrail[-1]) != key(subtrail[-2])
    """
    if minlen < 2:
        raise ValueError('The minimum length of a subtrail should be at least 2')

    if diffmode is None:
        for l in range(minlen, len(seq) + 1):
            yield seq[:l]
    elif diffmode == 'last':
        for l in range(minlen, len(seq) + 1):
            if diffkey(seq[l - 1]) != diffkey(seq[l - 2]):
                yield seq[:l]
    elif diffmode == 'all':
        for l in range(minlen, len(seq) + 1):
            if diffkey(seq[l - 1]) not in [diffkey(c) for c in seq[:(l - 1)]]:
                yield seq[:l]


def uniformed_trail_set(trl_set, key=lambda tr: tr[-1]['poi'], maxnum=1):
    """ Generating a trail set by prune the biased distribution of reference classes

        Arguments:
            trl_set: a set of trails

        Return:
            a set of trails that the last check-ins are of equal probability of being each class.
    """
    for poi, trls in itertools.groupby(sorted(list(trl_set), key=key), key=key):
        trls = list(trls)
        for _ in range(maxnum):
            yield trls[NP.random.randint(len(trls))]


def removed_duplication(seq, key=lambda ck: ck['poi']):
    """ Remove consecutive checkins at the same place

        Arguments:
            seq -- a trail

        Return:
            a trail with every consecutive checkins from different places
    """
    noduptrail = list()
    for c in seq:
        if len(noduptrail) == 0 or key(noduptrail[-1]) != key(c):
            noduptrail.append(c)
    return noduptrail


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
    def __init__(self, namespace, veclen=100, interval=(0., 24 * 3600.), kernel=gaussian_pdf, params=(3600,), isaccum=False, timeparser=TimeParser(), normalized=False):
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
            vec[idx][:] = kernel_smooth(self.axis, [self.timeparser.parse(c['tick']) for c in checkins], self.params, aggr=self.aggr, kernel=self.kernel)
        NP.add(vec, EPSILON, vec)
        if self.normalized:
            unity = NP.sum(vec, axis=0)
            NP.divide(vec, unity, vec)
        return vec

    def get_timeslot(self, t):
        """ Return the timeslot
        """
        return self.axis.searchsorted(self.timeparser.parse(t)) - 1
