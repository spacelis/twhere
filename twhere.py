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
import logging
import sys
import itertools

import numpy as NP

from model.colfilter import MemoryCFModel
from model.colfilter import CosineSimilarity
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
            self.parse = self.parse_cumulative
        else:
            self.parse = self.parse_binary

    def size(self):
        return len(self.namespace), 24 / self.unit

    def parse_binary(self, trail, istest=False):
        """ Parse a trail into a matrix
        """
        tmp = NP.zeros((len(self.namespace), 24 / self.unit), dtype=NP.float32)
        for c in trail:
            poi, t = self.columnfunc(c)
            tmp[poi, t] = 1
        if istest:
            tmp[:, t] = 0  # all checkins at time t will be removed for testing
            return tmp.flatten(), poi, t
        return tmp.flatten()

    def parse_cumulative(self, trail, istest=False):
        """ Parse a trail into a matrix
        """
        tmp = NP.zeros((len(self.namespace), 24 / self.unit), dtype=NP.float32)
        for c in trail:
            poi, t = self.columnfunc(c)
            tmp[poi, t] = tmp[poi, t] + 1
        if istest:
            tmp[:, t] = 0  # all checkins at time t will be removed for testing
            return tmp.flatten(), poi, t
        return tmp.flatten()

    def discretize(self, tick):
        return str(tick.hour / self.unit)

    def columnfunc(self, checkin):
        return self.namespace.index(checkin['poi']), self.discretize(checkin['tick'])


class CategoriesXContinuous(object):
    """ Parsing a sequence of check-ins into a staying matrix with each element
        indicating the number of times of check-ins.
    """
    def __init__(self, categories, div=100, sigmasquare=900. * 900.):
        self.namespace = [i for i in categories]
        self.div = div
        self.sigmasquare = sigmasquare

    def size(self):
        return len(self.namespace), self.div

    def parse(self, trail, istest=False):
        """ Parse a trail into a matrix
        """
        tmp = NP.zeros((len(self.namespace), self.div), dtype=NP.float32)
        if istest:
            prefix, lc = trail[:-1], trail[-1]
            poi, t = self.columnfunc(lc)
        else:
            prefix = trail
        prefix.sort(key=lambda ch: ch['poi'])
        for key, checkins in itertools.groupby(prefix, lambda ck: ck['poi']):
            tseq = list()
            for c in checkins:
                poi, t = self.columnfunc(c)
                tseq.append(t)
            tmp[poi] = self.gaussian_kernel(tseq)
        tmp = tmp.flatten()
        if istest:
            return tmp, poi, self.discretize(t)
        return tmp

    def discretize(self, tick):
        return int(tick / (3600 * 24) * self.div)

    def scale(self, tick):
        """
        """
        return float(tick.hour * 3600 + tick.minute * 60 + tick.second)

    def columnfunc(self, checkin):
        return self.namespace.index(checkin['poi']), self.scale(checkin['tick'])

    def gaussian_kernel(self, seq):
        """ Return the cumulative value of Gaussian PDF at each x in seq
        """
        return convoluted_gaussian(seq, self.sigmasquare, veclen=self.div, interval=(0., 24. * 3600))


class DiscreteTxC(object):
    """ A trail is treated as a set of (Place, Time) items representing a staying at a place
        and the time dimension is discrete as hours.
    """
    def __init__(self, parser, simnum=50):
        super(DiscreteTxC, self).__init__()
        self.parser = parser
        self.data = None
        self.model = None
        self.simnum = simnum

    def train(self, trails):
        """ Parse trails in to arrays so that they can be used in CF
        """
        data = list()
        for tr in trails:
            data.append(self.parser.parse(tr))
        self.data = NP.array(data, dtype=NP.float32)
        logging.info('%s values loaded in the model' % (self.data.shape,))
        self.model = MemoryCFModel(self.data, CosineSimilarity, LinearCombination)
        #self.model = MemoryCFModel(self.data, JaccardSimilarity_GPU, LinearCombination)

    def predict(self, trail, time):
        """ predict a trail of stay at `time`
        """
        est = self.model.estimate(trail, num=self.simnum)
        p = est.reshape(self.parser.size())[:, time]
        return [self.parser.namespace[i] for i in NP.argsort(p)][::-1]

    def evaluate(self, trails):
        """ evaluate the model using a set of `trails`
        """
        result = NP.zeros(len(trails), dtype=NP.int32)
        for idx, tr in enumerate(trails):
            with NP.errstate(divide='raise'):
                try:
                    ftr, ref, t = self.parser.parse(tr, istest=True)
                    rank = self.predict(ftr, t)
                    result[idx] = rank.index(self.parser.namespace[ref]) + 1
                except FloatingPointError:
                    logging.warn('%s has no similar trails.', (tr[0]['trail_id']))
                    result[idx] = len(self.parser.namespace)
        return result


def experiment():
    """ running the experiment
    """
    city = sys.argv[1]
    poicol = 'category'
    data_provider = TextData(city, poicol)
    logging.info('Reading data from %s', city)
    data = data_provider.get_data()
    logging.info('Predicting %s', poicol)
    # Parser parameters @parser
    parser = CategoriesXContinuous(data_provider.get_namespace(), div=400, sigmasquare=600.*600.)
    for trainset, testset in cv_splites(data, len(data)):
        m = DiscreteTxC(parser)
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
