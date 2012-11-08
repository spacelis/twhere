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

import sys
import logging
logging.basicConfig(format='%(asctime)s %(name)s [%(levelname)s] %(message)s', level=logging.INFO)
LOGGER = logging.getLogger(__name__)

import site
site.addsitedir('../model')

from random import shuffle

import numpy as NP
NP.seterr(all='warn', under='ignore')

from model.colfilter import VectorDatabase
from trail import TrailGen
from trail import TrailSet
from trail import KernelVectorizor
from trail import TimeParser
from dataprov import TextData
from experiment.crossvalid import folds
from model.colfilter import CosineSimilarity
from model.colfilter import LinearCombination


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

    def evaluate(self, tr):
        """ predicting the last
        """
        rank = list(self.namespace)
        shuffle(rank)
        pos = rank.index(self.majority)
        rank[0], rank[pos] = rank[pos], rank[0]
        result = rank.index(tr[-1]['poi']) + 1
        return result


class PredictingLast(object):
    """ Always predicting the last
    """
    def __init__(self, namespace):
        super(PredictingLast, self).__init__()
        self.namespace = namespace

    def train(self, trails):
        pass

    def evaluate(self, tr):
        """ predicting the last
        """
        rank = list(self.namespace)
        shuffle(rank)
        pos = rank.index(tr[-2]['poi'])
        rank[0], rank[pos] = rank[pos], rank[0]
        result = rank.index(tr[-1]['poi']) + 1
        return result


class ColfilterModel(object):
    """ Using cofilter with vectorized trails
    """
    def __init__(self, namespace, vdb_conf, krl_conf):
        """ Constructor

            Arguments:
                namespace -- the predicting classes
                veclen -- the length of vectors
                vdb_conf -- a string for configuring VectorDatabase, e.g., '{"simnum": 20, "similarity": CosineSimilarity(), "aggregator": LinearCombination()}'
                krl_conf -- a string representing kernel used, e.g. '{"veclen": 100, "interval": (0., 24*3600.), "kernel": "gaussian", "params": (3600.,), "isaccum": False, "normalized": False}'
        """
        super(ColfilterModel, self).__init__()
        LOGGER.debug('VDBCONF=' + vdb_conf)
        LOGGER.debug('KRLCONF=' + krl_conf)
        self.namespace = namespace
        self.timeparser = TimeParser()
        # Prepare model with kernel specification
        kparams = eval(krl_conf)
        kparams["namespace"] = namespace
        self.vectorizor = KernelVectorizor(**kparams)
        vparams = eval(vdb_conf)
        self.model = VectorDatabase(**vparams)

    def train(self, trails):
        """ Train the model
        """
        self.vecs = NP.zeros((len(trails), len(self.namespace), self.vectorizor.veclen), dtype=NP.float32)
        for idx, tr in enumerate(trails):
            self.vecs[idx, :, :] = self.vectorizor.process(tr)
        self.model.load_data(self.vecs)

    def evaluate(self, tr):
        """ Predicting with colfilter model
        """
        history = tr[:-1]
        refpoi, t = tr[-1]['poi'], self.vectorizor.get_timeslot(tr[-1]['tick'])
        vec = self.vectorizor.process(history)
        est = self.model.estimates(vec, t)
        rank = list(reversed(NP.argsort(est[:, t])))
        result = rank.index(self.namespace.index(refpoi)) + 1      # rank should start from 1 because to MRR
        return result


def experiment():
    """ running the experiment
    """
    # Parameters
    if len(sys.argv) < 3:
        LOGGER.error('Insufficient number of parameters')
        LOGGER.info('Usage: twhere.py <city-id> {category|base} <model> [<vdb_str> <krl_str>]')
        sys.exit(-1)
    city = sys.argv[1]
    poicol = sys.argv[2]
    modelname = sys.argv[3]
    if modelname == 'colfilter':
        vdb_str = sys.argv[4]  # see ColfilterModel()
        krl_str = sys.argv[5]
    data_provider = TextData(city, poicol)

    # Experiment
    LOGGER.info('Reading data from %s', city)
    data = data_provider.get_data()
    LOGGER.info('Predicting %s', poicol)
    for fold_id, (testset, trainset) in enumerate(folds(data, 10)):
        LOGGER.info('Fold: %d/%d' % (fold_id + 1, 10))
        if modelname == 'colfilter':
            m = ColfilterModel(data_provider.get_namespace(), vdb_str, krl_str)
        elif modelname == 'last':
            m = PredictingLast(data_provider.get_namespace())
        elif modelname == 'major':
            m = PredictingMajority(data_provider.get_namespace())
        else:
            LOGGER.error('The model specified does not exist: %s' % (modelname, ))
            LOGGER.info('colfilter, last, major')
            sys.exit(-2)

        LOGGER.info('Training...')
        train_tr = [tr for tr in TrailGen(trainset, lambda x:x['trail_id'])]
        test_tr = [tr for tr in TrailGen(testset, lambda x:x['trail_id']) if len(tr) > 5]
        m.train(train_tr)
        LOGGER.info('Checkins: %d / %d' % (sum(map(len, test_tr)), sum(map(len, train_tr))))
        LOGGER.info('Trails: ' + str(len(test_tr)) + ' / ' + str(len(train_tr)))
        LOGGER.info('Testing...')
        for trail in test_tr:
            print m.evaluate(trail)
            LOGGER.debug('Trail ID: ' + trail[0]['trail_id'])
            #for tr in TrailSet(trail, 3):
                #print m.evaluate(tr)
        sys.exit(0)


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

    m = ColfilterModel(['home', 'ewi', 'canteen'], 24, '20|CosineSimilarity()|LinearCombination()', '(0., 24*3600.)|gaussian|(3600.,)')
    LOGGER.info('Training...')
    m.train([tr for tr in TrailGen(trainset, lambda x:x['trail_id'])])
    LOGGER.info('Testing...')
    for e in m.evaluate([tr for tr in TrailGen(testset, lambda x:x['trail_id'])]):
        print e
        assert_equal(e, 1)


if __name__ == '__main__':
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (2.5 * 1024 * 1024 * 1024L, -1L))
    except:
        LOGGER.warn('Failed set resource limits.')

    LOGGER.debug('DEBUG is enabled')
    #experiment()
    import profile
    profile.run('experiment()')
