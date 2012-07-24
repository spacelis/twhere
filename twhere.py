#!/bin/python
# -*- coding: utf-8 -*-
"""File: trailprediction.py
Description:
    Predict users trail by their previous visits.
History:
    0.1.0 The first version.
"""
__version__ = '0.1.0'
__author__ = 'SpaceLis'
import site
site.addsitedir('../model')
import resource
resource.setrlimit(resource.RLIMIT_AS, (2.5 * 1024 * 1024 * 1024L, -1L))

import logging
import sys
import itertools

import numpy as NP
NP.seterr(all='warn')

from model.colfilter import MemoryCFModel
from model.colfilter import CosineSimilarity
from model.colfilter import CosineSimilarityDamping
from model.colfilter import LinearCombination
from model.colfilter import convoluted_gaussian
from trail import TrailGen
from dataprov import TextData
from crossvalid import cv_splites


class CategoriesX24h(object):
    """ Parsing a sequence of check-ins into a staying matrix with each element
        indicating whether user stayed at the place at the time.
    """
    def __init__(self, categories, unit=1, iscumulative=False):
        self.namespace = [i for i in categories]
        self.unit = unit
        if iscumulative:
            self.vectorize = self.vectorize_cumulative
        else:
            self.vectorize = self.vectorize_binary

    def size(self):
        return len(self.namespace), 24 / self.unit

    def vectorize_binary(self, trail, istest=False):
        """ Parse a trail into a matrix
        """
        tmp = NP.zeros((len(self.namespace), 24 / self.unit), dtype=NP.float32)
        for c in trail:
            poi, t = self.parse_checkin(c)
            tmp[poi, self.discretize(t)] = 1
        if istest:
            tmp[:, self.discretize(t)] = 0  # all checkins at time t will be removed for testing
            return tmp, poi, t
        return tmp

    def vectorize_cumulative(self, trail, istest=False):
        """ Parse a trail into a matrix
        """
        tmp = NP.zeros((len(self.namespace), 24 / self.unit), dtype=NP.float32)
        for c in trail:
            poi, t = self.parse_checkin(c)
            dist = self.discretize(t)
            tmp[poi, dist] = tmp[poi, dist] + 1
        if istest:
            tmp[:, dist] = 0  # all checkins at time t will be removed for testing
            return tmp, poi, t
        return tmp

    def discretize(self, tick):
        return tick / self.unit

    def normalize(self, dt_obj):
        return dt_obj.hour

    def parse_checkin(self, checkin):
        return self.namespace.index(checkin['poi']), self.normalize(checkin['tick'])


class CategoriesXContinuous(object):
    """ Parsing a sequence of check-ins into a staying matrix with each element
        indicating the number of times of check-ins.
    """
    def __init__(self, categories, div=100, sigma=1800.):
        self.namespace = [i for i in categories]
        self.div = div
        self.sigmasquare = float(sigma * sigma)

    def size(self):
        return len(self.namespace), self.div

    def vectorize(self, trail, istest=False):
        """ Parse a trail into a matrix
        """
        tmp = NP.zeros((len(self.namespace), self.div), dtype=NP.float32)
        if istest:
            prefix, lc = trail[:-1], trail[-1]
            poi, t = self.parse_checkin(lc)
        else:
            prefix = trail
        prefix.sort(key=lambda ch: ch['poi'])
        for key, checkins in itertools.groupby(prefix, lambda ck: ck['poi']):
            tseq = list()
            for c in checkins:
                poi, t = self.parse_checkin(c)
                tseq.append(t)
            tmp[poi] = self.gaussian_kernel(tseq)
        tmp = tmp
        if istest:
            return tmp, poi, t
        return tmp

    def discretize(self, tick):
        """ Discretization time in to segments for vectorizing
        """
        return int(tick / (3600 * 24) * self.div)

    def normalize(self, tick):
        """ Converting to seconds
        """
        return float(tick.hour * 3600 + tick.minute * 60 + tick.second)

    def parse_checkin(self, checkin):
        """ Extract and convert related information in a check-in into internal types
        """
        return self.namespace.index(checkin['poi']), self.normalize(checkin['tick'])

    def gaussian_kernel(self, seq):
        """ Return the cumulative value of Gaussian PDF at each x in seq
        """
        return convoluted_gaussian(seq, self.sigmasquare, veclen=self.div, interval=(0., 24. * 3600))


class Baseline(object):
    """ Always predicting the last
    """
    def __init__(self, parser):
        super(Baseline, self).__init__()
        self.parser = parser

    def train(self, trails):
        pass

    def evaluate(self, trails):
        """ predicting the last
        """
        from random import shuffle
        result = NP.zeros(len(trails), dtype=NP.int32)
        for idx, tr in enumerate(trails):
            rank = list(self.parser.namespace)
            shuffle(rank)
            pos = rank.index(tr[-2]['poi'])
            rank[0], rank[pos] = rank[pos], rank[0]
            result[idx] = rank.index(tr[-1]['poi']) + 1
        return result


class DiscreteTxC(object):
    """ A trail is treated as a set of (Place, Time) items representing a staying at a place
        and the time dimension is discrete as hours.
    """
    def __init__(self, parser, simfunc, combfunc, simnum=50, **kargs):
        super(DiscreteTxC, self).__init__()
        self.parser = parser
        self.simfunc = simfunc
        self.combfunc = combfunc
        self.data = None
        self.model = None
        self.simnum = simnum
        self.subkargs = kargs

    def train(self, trails):
        """ Parse trails in to arrays so that they can be used in CF
        """
        data = list()
        for tr in trails:
            data.append(self.parser.vectorize(tr))
        self.data = NP.array(data, dtype=NP.float32)
        logging.info('%s values loaded from data source', self.data.shape)
        self.model = MemoryCFModel(self.data, self.simfunc, self.combfunc)

    def predict(self, trail, tick):
        """ predict a trail of stay at `tick`
        """
        est = self.model.estimate(trail, num=self.simnum, tick=tick)
        p = est[:, self.parser.discretize(tick)]
        return [self.parser.namespace[i] for i in NP.argsort(p)][::-1]

    def evaluate(self, trails):
        """ evaluate the model using a set of `trails`
        """
        result = NP.zeros(len(trails), dtype=NP.int32)
        for idx, tr in enumerate(trails):
            with NP.errstate(divide='raise'):
                try:
                    ftr, ref, t = self.parser.vectorize(tr, istest=True)
                    rank = self.predict(ftr, t)
                    result[idx] = rank.index(self.parser.namespace[ref]) + 1
                except FloatingPointError:
                    logging.warn('%s has no similar trails.', tr[0]['trail_id'])
                    result[idx] = len(self.parser.namespace)
        logging.info('%s trails are tested', len(result))
        return result


def experiment():
    """ running the experiment
    """
    # Parameters
    city = sys.argv[1]
    simnum = int(sys.argv[2])
    sigma = float(sys.argv[3])
    poicol = 'category'
    data_provider = TextData(city, poicol)
    veclen = 100

    # Experiment
    logging.info('Reading data from %s', city)
    data = data_provider.get_data()
    logging.info('Predicting %s', poicol)
    parser = CategoriesXContinuous(data_provider.get_namespace(), div=veclen, sigma=sigma)
    for trainset, testset in cv_splites(data, len(data)):
        #m = DiscreteTxC(parser, CosineSimilarity(), LinearCombination(), simnum=simnum)
        m = DiscreteTxC(parser, CosineSimilarityDamping(factor=float(sys.argv[4])), LinearCombination(), simnum=simnum)
        #m = Baseline(parser)
        logging.info('Training...')
        m.train([tr for tr in TrailGen(trainset, lambda x:x['trail_id'])])
        logging.info('Testing...')
        for e in m.evaluate([tr for tr in TrailGen(testset, lambda x:x['trail_id']) if len(tr) > 5]):
            print e


def test_model():
    """ Test the model
    """
    from datetime import datetime
    column = ['trail_id', 'poi', 'tick', 'text']
    train = [['1', 'home', datetime(2011, 1, 1, 6, 30), 'test'],
            ['1', 'ewi', datetime(2011, 1, 1, 9, 30), 'test'],
            ['1', 'canteen', datetime(2011, 1, 1, 12, 30), 'test'],
            ['1', 'ewi', datetime(2011, 1, 1, 14, 30), 'test'],

            ['2', 'ewi', datetime(2011, 1, 1, 11, 30), 'test'],
            ['2', 'canteen', datetime(2011, 1, 1, 12, 30), 'test'],
            ['2', 'ewi', datetime(2011, 1, 1, 14, 30), 'test'],
            ['2', 'home', datetime(2011, 1, 1, 18, 30), 'test']]

    trainset = [dict([(key, val) for key, val in zip(column, checkin)]) for checkin in train]

    test = [['3', 'home', datetime(2011, 1, 1, 6, 30), 'test'],
            ['3', 'ewi', datetime(2011, 1, 1, 9, 30), 'test'],
            ['3', 'canteen', datetime(2011, 1, 1, 12), 'test'],
            ['3', 'ewi', datetime(2011, 1, 1, 14, 25), 'test']]

    testset = [dict([(key, val) for key, val in zip(column, checkin)]) for checkin in test]

    parser = CategoriesXContinuous(['ewi', 'home', 'canteen'])
    #parser = CategoriesX24h(['ewi', 'home', 'canteen'])
    m = DiscreteTxC(parser)
    logging.info('Training...')
    m.train([tr for tr in TrailGen(trainset, lambda x:x['trail_id'])])
    logging.info('Testing...')
    for e in m.evaluate([tr for tr in TrailGen(testset, lambda x:x['trail_id'])]):
        print e


def test():
    """ Test
    """
    #test_model()
    experiment()

if __name__ == '__main__':
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)
    test()
