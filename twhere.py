#!/usr/bin/env python
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
logging.basicConfig(format='%(asctime)s %(name)s [%(levelname)s] %(message)s', level=logging.INFO)
LOGGER = logging.getLogger(__name__)

from random import shuffle

import numpy as NP
NP.seterr(all='warn', under='ignore')

from model.colfilter import VectorDatabase
from model.mm import MarkovModel
from trail import itertrails
from trail import iter_subtrails
from trail import KernelVectorizor
from trail import TimeParser
from dataprov import TextData
from experiment.crossvalid import folds
import config


class PredictingMajority(object):
    """ Preducting the majority class
    """
    def __init__(self, namespace):
        super(PredictingMajority, self).__init__()
        self.namespace = namespace
        self.dist = dict([(n, 0) for n in namespace])

    def train(self, trails):
        """ Calculate the distribution of visiting location type
        """
        for tr in trails:
            for c in tr:
                self.dist[c['poi']] += 1
        self.majority = sorted(self.dist.iteritems(), key=lambda x: x[1], reverse=True)[0][0]
        LOGGER.info('Majority Class: %s' % (self.majority,))

    def predict(self, tr, tick):
        """ predicting the last
        """
        rank = list(self.namespace)
        shuffle(rank)
        pos = rank.index(self.majority)
        rank[0], rank[pos] = rank[pos], rank[0]
        return rank


class PredictingLast(object):
    """ Always predicting the last
    """
    def __init__(self, namespace):
        super(PredictingLast, self).__init__()
        self.namespace = namespace

    def train(self, trails):
        pass

    def predict(self, tr, tick):
        """ predicting the last
        """
        rank = list(self.namespace)
        shuffle(rank)
        pos = rank.index(tr[-1]['poi'])
        rank[0], rank[pos] = rank[pos], rank[0]
        return rank


class MarkovChainModel(object):
    """ Using Markov Model for estimation
    """
    def __init__(self, namespace):
        super(MarkovChainModel, self).__init__()
        self.namespace = namespace
        self.model = MarkovModel(self.namespace)

    def train(self, trails):
        """ Train the model
        """
        poitrails = list()
        for tr in trails:
            poitrails.append([c['poi'] for c in tr])
        self.model.learn_from(poitrails)

    def predict(self, tr, tick):
        """ Evaluate the model by trails
        """
        return self.model.rank_next(tr[-1]['poi'])


class ColfilterModel(object):
    """ Using cofilter with vectorized trails
    """
    def __init__(self, namespace):
        """ Constructor

            Arguments:
                namespace -- the predicting classes
                veclen -- the length of vectors
        """
        super(ColfilterModel, self).__init__()
        self.namespace = namespace
        self.mapping = dict([(poi, idx) for idx, poi in enumerate(self.namespace)]).get
        self.timeparser = TimeParser()
        # Prepare model with kernel specification
        kparams = dict(config.VECTORIZOR_PARAM)
        kparams["namespace"] = namespace
        self.vectorizor = KernelVectorizor(**kparams)
        self.model = VectorDatabase(**config.VECTORDB_PARAM)

    def train(self, trails):
        """ Train the model
        """
        self.vecs = NP.zeros((len(trails), len(self.namespace), self.vectorizor.veclen), dtype=NP.float32)
        for idx, tr in enumerate(trails):
            self.vecs[idx, :, :] = self.vectorizor.process(tr)
        self.model.load_data(self.vecs)

    def predict(self, tr, tick):
        """ Predicting with colfilter model
        """
        t = self.vectorizor.get_timeslot(tick)
        vec = self.vectorizor.process(tr)
        est = self.model.estimates(vec, t)
        rank = sorted(self.namespace, key=lambda x: est[self.mapping(x), t], reverse=True)
        return rank


def run_test(model, trail):
    """ running the testing stage
    """
    history, ref = trail[:-1], trail[-1]
    rank = model.predict(history, ref['tick'])
    return rank.index(ref['poi']) + 1


def print_trail(trail):
    """ printing trail in a compact way
    """
    pois = [c['poi'][-8:-4] for c in trail]
    ticks = [c['tick'].strftime('%H:%M:%S') for c in trail]
    print zip(pois, ticks)


FOLDS = 5


def run_experiment(city, poicol, model, fname):
    """ running the experiment
    """
    LOGGER.info('Reading data from %s', city)
    data_provider = TextData(city, poicol)
    data = data_provider.get_data()
    output = open(fname, 'w')

    LOGGER.info('Predicting %s', poicol)
    for fold_id, (testset, trainset) in enumerate(folds(data, FOLDS)):
        LOGGER.info('Fold: %d/%d' % (fold_id + 1, FOLDS))
        m = model(data_provider.get_namespace())

        LOGGER.info('Training...')
        train_tr = [tr for tr in itertrails(trainset, lambda x:x['trail_id'])]
        test_tr = [tr for tr in itertrails(testset, lambda x:x['trail_id'])]
        m.train(train_tr)

        LOGGER.info('Checkins: %d / %d' % (sum(map(len, test_tr)), sum(map(len, train_tr))))
        LOGGER.info('Trails: ' + str(len(test_tr)) + ' / ' + str(len(train_tr)))
        LOGGER.info('Testing...[Output: %s]' % (output.name,))

        test_cnt = 0
        for trail in test_tr:
            for subtrl in iter_subtrails(trail, minlen=5, diffkey=lambda x: x['poi']):
                print >> output, run_test(m, subtrl)
                test_cnt += 1
        LOGGER.info('Tested trails: %d' % (test_cnt,))


def test_model():
    """ Test the model
    """
    from datetime import datetime
    from numpy.testing import assert_equal
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

    #FIXME should be changed according to new interface
    m = ColfilterModel(['home', 'ewi', 'canteen'], "{'simnum': 24, 'similarity': CosineSimilarity(), 'aggregator': LinearCombination()}", "{'veclen': 100, 'interval': (0.,24*3600.), 'kernel': 'gaussian', 'params': (3600.,), 'isaccum': True}")
    LOGGER.info('Training...')
    m.train([tr for tr in itertrails(trainset, lambda x:x['trail_id'])])
    LOGGER.info('Testing...')
    for tr in itertrails(testset, lambda x: x['trail_id']):
        e = m.evaluate(tr)
        print e
        assert_equal(e, 1)


if __name__ == '__main__':
    raise Exception('Should run experiment.py instead of this one')
