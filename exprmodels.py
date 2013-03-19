#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""File: trailprediction.py
Description:
    Predict users trail by their previous visits.
History:
    0.2.0 Long trail scheme
    0.1.0 The first version.
"""
__version__ = '0.2.0'
__author__ = 'SpaceLis'

import logging
from random import shuffle
from datetime import timedelta
from collections import Counter
import resource

import numpy as NP
NP.seterr(all='warn', under='ignore')

from mlmodels.experiment.crossvalid import folds
from mlmodels.model.colfilter import SparseVectorDatabase
from mlmodels.model.mm import MarkovModel
from twhere.trail import checkin_trails
from twhere.trail import KernelVectorizor
from twhere.trail import as_segments
from twhere.dataprov import TextData


class PredictingMajority(object):
    """ Preducting the majority class
    """
    def __init__(self, conf):
        super(PredictingMajority, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.namespace = conf['data.namespace']
        self.dist = Counter(self.namespace)
        self.majority = None

    def train(self, trail_set):
        """ Calculate the distribution of visiting location type
        """
        for tr in trail_set:
            self.dist.update([c['poi'] for c in tr])
        self.majority = [x[0] for x in self.dist.most_common()]
        self.logger.info('Majority Class: %s' % (self.majority[0],))

    def predict(self, tr, tick): # pylint: disable-msg=W0613
        """ predicting the last
        """
        return self.majority


class PredictingTimeMajority(object):
    """ Preducting the majority class
    """
    def __init__(self, conf):
        super(PredictingTimeMajority, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.namespace = conf['data.namespace']
        self.seglen = conf['cf.segment']
        self.mapping = dict([(poi, idx) for idx, poi in enumerate(self.namespace)]).get
        # Prepare model with kernel specification
        self.vectorizor = KernelVectorizor.from_config(conf)
        self.dist = NP.zeros((len(self.namespace), self.seglen))

    def train(self, trail_set):
        """ Calculate the distribution of visiting location type
        """
        for tr in trail_set:
            for c in tr:
                t = self.vectorizor.get_timeslot(c['tick']) % self.seglen
                p = self.mapping(c['poi'])
                self.dist[p, t] += 1

    def predict(self, tr, tick):  # pylint: disable-msg=W0613
        """ predicting the last
        """
        return sorted(self.namespace,
                      key=lambda x: self.dist[self.mapping(x), self.vectorizor.get_timeslot(tick) % self.seglen],
                      reverse=True)


class PredictingLast(object):
    """ Always predicting the last
    """
    def __init__(self, conf):
        super(PredictingLast, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.namespace = conf['data.namespace']

    def train(self, trail_set):
        """ Train model
        """
        pass

    def predict(self, tr, tick):  # pylint: disable-msg=W0613
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
    def __init__(self, conf):
        super(MarkovChainModel, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.namespace = conf['data.namespace']
        self.model = MarkovModel(self.namespace)

    def train(self, trail_set):
        """ Train the model
        """
        poitrails = list()
        for tr in trail_set:
            poitrails.append([c['poi'] for c in tr])
        self.model.learn_from(poitrails)

    def predict(self, tr, tick):  # pylint: disable-msg=W0613
        """ Evaluate the model by trail_set
        """
        return self.model.rank_next(tr[-1]['poi'])


class ColfilterModel(object):
    """ Using cofilter with vectorized trail_set
    """
    def __init__(self, conf):
        """ Constructor

            Arguments:
                conf -- a dict-like object holding configurations
        """
        super(ColfilterModel, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.namespace = conf['data.namespace']
        self.seglen = conf['cf.segment']
        self.mapping = dict([(poi, idx) for idx, poi in enumerate(self.namespace)]).get

        # Prepare model with kernel specification
        self.vectorizor = KernelVectorizor.from_config(conf)
        self.model = SparseVectorDatabase.from_config(conf)
        self.vecs = None
        self.ck_cnt = None

    def train(self, trail_set):
        """ Train the model
        """
        for _, tr in enumerate(trail_set):
            vecs = as_segments(self.vectorizor.process(tr), self.seglen)
            self.model.extend_dataitems(vecs)
        self.logger.info('Resource usage: {0}MB'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000))

    def predict(self, tr, tick):
        """ Predicting with colfilter model
        """
        t = self.vectorizor.get_timeslot(tick)
        vec = self.vectorizor.process(tr)[:, t - self.seglen + 1: t + 1]

        est = self.model.estimates(vec, t)
        rank = sorted(self.namespace, key=lambda x: est[self.mapping(x), -1], reverse=True)
        return rank


def rank_ref(model, history, reftick, refpoi):
    """ running the testing stage
    """
    rank = model.predict(history, reftick)
    return rank.index(refpoi) + 1


def iter_test_instance(test_tr_set, segperiod=timedelta(hours=24)):
    """ generating testing instance
    """
    for trl in test_tr_set:
        trl = sorted(trl, key=lambda x: x['tick'])
        for ref_checkin in trl:
            segend = ref_checkin['tick']
            segstart = segend - segperiod
            segtrail = [c for c in trl if segstart < c['tick'] < segend]
            if len(segtrail) < 2:
                continue
            yield segtrail


def print_trail(trail):
    """ printing trail in a compact way
    """
    pois = [c['poi'][-8:-4] for c in trail]
    ticks = [c['tick'].strftime('%H:%M:%S') for c in trail]
    print zip(pois, ticks)


FOLDS = 10


def experiment(conf): # pylint: disable-msg=R0914
    """ running the experiment
    """
    logger = logging.getLogger(__name__)
    logger.info('--------------------  Experimenting on {0}'.format(conf['expr.city.name']))
    logger.info('Reading data from {0}'.format(conf['expr.city.name']))
    data_provider = TextData(conf['expr.city.id'], conf['expr.target'])
    conf['data.namespace'] = data_provider.get_namespace()
    data = data_provider.get_data()
    output = open(conf['expr.output'], 'w')
    model = globals()[conf['expr.model']]

    logger.info('Predicting {0}'.format(conf['expr.target']))
    total_trails = checkin_trails(data)
    logger.info('Trails in given dataset: {0}'.format(len(total_trails)))
    for fold_id, (test_tr, train_tr) in enumerate(folds(total_trails, FOLDS)):
        logger.info('----------  Fold: {0}/{1}'.format(fold_id + 1, FOLDS))

        logger.info('Training...')
        m = model(conf)
        train_tr_set = list(train_tr)
        test_tr_set = list(test_tr)
        m.train(train_tr_set)

        logger.info('Checkins: {0} / {1}'.format(sum([len(tr) for tr in test_tr_set]),
                                                 sum([len(tr) for tr in train_tr_set])))
        logger.info('Trails: {0} / {1}'.format(len(test_tr_set), len(train_tr_set)))
        logger.info('Testing...[Output: {0}]'.format(output.name))

        test_cnt = Counter()
        for segtrl in iter_test_instance(test_tr_set):
            htrl, reftick, refpoi = segtrl[:-1], segtrl[-1]['tick'], segtrl[-1]['poi']
            print >> output, rank_ref(m, htrl, reftick, refpoi)
            test_cnt.update(['instances'])
        logger.info('Tested trails: {0}'.format(test_cnt['instances']))


if __name__ == '__main__':
    raise Exception('Should run experiment.py instead of this one')
