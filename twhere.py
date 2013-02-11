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

from model.colfilter import VectorDatabase, uniform_pdf
from model.mm import MarkovModel
from trail import itertrails
from trail import iter_subtrails
from trail import KernelVectorizor
from trail import TimeParser
from dataprov import TextData
from experiment.crossvalid import folds
from collections import Counter
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


class PredictingTimeMajority(object):
    """ Preducting the majority class
    """
    def __init__(self, namespace):
        super(PredictingTimeMajority, self).__init__()
        self.namespace = namespace
        self.mapping = dict([(poi, idx) for idx, poi in enumerate(self.namespace)]).get
        self.timeparser = TimeParser()
        # Prepare model with kernel specification
        kparams = dict(config.VECTORIZOR_PARAM)
        kparams["namespace"] = namespace
        self.vectorizor = KernelVectorizor(**kparams)
        self.dist = NP.zeros((len(namespace), config.VECTORIZOR_PARAM['veclen']))

    def train(self, trails):
        """ Calculate the distribution of visiting location type
        """
        for tr in trails:
            for c in tr:
                t = self.vectorizor.get_timeslot(c['tick'])
                p = self.mapping(c['poi'])
                self.dist[p, t] += 1
        LOGGER.info('Majority Class: %s' % (self.dist,))

    def predict(self, tr, tick):
        """ predicting the last
        """
        return sorted(self.namespace, key=lambda x: self.dist[self.mapping(x), self.vectorizor.get_timeslot(tick)], reverse=True)


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


class ColfilterHistoryModel(object):
    """ Using cofilter with vectorized trails but rank category by most visits in similar trails
    """
    def __init__(self, namespace):
        """ Constructor

            Arguments:
                namespace -- the predicting classes
                veclen -- the length of vectors
        """
        super(ColfilterHistoryModel, self).__init__()
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
        self.recs = list()
        for idx, tr in enumerate(trails):
            self.vecs[idx, :, :] = self.vectorizor.process(tr)
            self.recs.append(Counter([c['poi'] for c in tr]))
        self.model.load_data(self.vecs)

    def predict(self, tr, tick):
        """ Predicting with colfilter model
        """
        t = self.vectorizor.get_timeslot(tick)
        vec = self.vectorizor.process(tr)
        sims = self.model.get_similarities(vec, t)
        idces = NP.argsort(sims)[::-1]
        dist = NP.zeros(len(self.namespace))
        for idx in idces[:config.VECTORDB_PARAM['simnum']]:
            for n, cnt in self.recs[idx].iteritems():
                dist[self.mapping(n)] += cnt * sims[idx]
        rank = sorted(self.namespace, key=lambda x: dist[self.mapping(x)], reverse=True)
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


FOLDS = 10


def run_experiment(city, poicol, model, fname):
    """ running the experiment
    """
    LOGGER.info('Reading data from %s', city)
    data_provider = TextData(city, poicol)
    data = data_provider.get_data()
    output = open(fname, 'w')

    LOGGER.info('Predicting %s', poicol)
    total_trails = list(itertrails(data, lambda x: x['trail_id']))
    LOGGER.info('Trails in given dataset: %s', len(total_trails))
    for fold_id, (test_tr, train_tr) in enumerate(folds(total_trails, FOLDS)):
        LOGGER.info('Fold: %d/%d' % (fold_id + 1, FOLDS))

        LOGGER.info('Training...')
        m = model(data_provider.get_namespace())
        train_tr = list(train_tr)
        test_tr = list(test_tr)
        m.train(train_tr)

        LOGGER.info('Checkins: %d / %d' % (sum(map(len, test_tr)), sum(map(len, train_tr))))
        LOGGER.info('Trails: ' + str(len(test_tr)) + ' / ' + str(len(train_tr)))
        LOGGER.info('Testing...[Output: %s]' % (output.name,))

        test_cnt = 0
        for trail in test_tr:
            for subtrl in iter_subtrails(trail, minlen=5, diffkey=lambda x: x['pid']):
                print >> output, run_test(m, subtrl)
                test_cnt += 1
        LOGGER.info('Tested trails: %d' % (test_cnt,))


if __name__ == '__main__':
    raise Exception('Should run experiment.py instead of this one')
