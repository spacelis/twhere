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

import os
import logging
from datetime import timedelta
from collections import Counter
import resource
from random import shuffle

import numpy as NP
import bottleneck as BN
NP.seterr(all='warn', under='ignore')
from numpy.lib.stride_tricks import as_strided

from mlmodels.experiment.crossvalid import folds
from mlmodels.model.colfilter import SparseVectorDatabase
from mlmodels.model.colfilter import VectorDatabase
from mlmodels.model.mm import MarkovModel
from twhere.trail import checkin_trails
from twhere.trail import KernelVectorizor
from twhere.trail import as_doublesegments
from twhere.trail import as_mask
from twhere.trail import as_vector_segments
from twhere.trail import diff_last_filter  # pylint: disable-msg=W0611
from twhere.trail import diff_all_filter  # pylint: disable-msg=W0611
from twhere.trail import length_limits_filter  # pylint: disable-msg=W0611
from twhere.dataprov import TextData
from twhere.beeper import Beeper


class PredictingMajority(object):
    """ Preducting the majority class
    """
    def __init__(self, conf):
        super(PredictingMajority, self).__init__()
        self.logger = logging.getLogger(
            '%s.%s' % (__name__, type(self).__name__))
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

    def predict(self, tr, tick):  # pylint: disable-msg=W0613
        """ predicting the last
        """
        return self.majority


class PredictingTimeMajority(object):
    """ Preducting the majority class
    """
    def __init__(self, conf):
        super(PredictingTimeMajority, self).__init__()
        self.logger = logging.getLogger(
            '%s.%s' % (__name__, type(self).__name__))
        self.namespace = conf['data.namespace']
        self.seglen = conf['cf.segment']
        self.mapping = dict(
            [(poi, idx) for idx, poi in enumerate(self.namespace)]
        ).get
        # Prepare model with kernel specification
        self.vectorizor = KernelVectorizor.from_config(conf)
        self.dist = NP.zeros(
            (len(self.namespace), self.seglen), dtype=NP.int32)

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
        t = self.vectorizor.get_timeslot(tick) % self.seglen
        return sorted(self.namespace,
                      key=lambda x: self.dist[self.mapping(x), t],
                      reverse=True)


class RandomGuess(object):
    """ Random guess
    """
    def __init__(self, conf):
        super(RandomGuess, self).__init__()
        self.namespace = conf['data.namespace']

    def train(self, trail_set):
        """ Train model
        """
        pass

    def predict(self, tr, tick):  # pylint: disable-msg=W0613
        """ Predicting the check-in at tick
        """
        rank = list(self.namespace)
        shuffle(rank)
        return rank


class PredictingLast(object):
    """ Always predicting the last
    """
    def __init__(self, conf):
        super(PredictingLast, self).__init__()
        self.logger = logging.getLogger(
            '%s.%s' % (__name__, type(self).__name__))
        self.namespace = conf['data.namespace']
        self.fallback = globals()[conf['predlast.fallback']](conf)

    def train(self, trail_set):
        """ Train model
        """
        self.fallback.train(trail_set)

    def predict(self, tr, tick):
        """ predicting the last
        """
        rank = self.fallback.predict(tr, tick)
        pos = rank.index(tr[-1]['poi'])
        rank.insert(0, rank.pop(pos))
        return rank


class MarkovChainModel(object):
    """ Using Markov Model for estimation
    """
    def __init__(self, conf):
        super(MarkovChainModel, self).__init__()
        self.logger = logging.getLogger(
            '%s.%s' % (__name__, type(self).__name__))
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


class SimpleColfilterModel(object):
    """ A simple colfilter model without considering time
    """
    def __init__(self, conf):
        super(SimpleColfilterModel, self).__init__()
        self.logger = logging.getLogger(
            '%s.%s' % (__name__, type(self).__name__))
        self.namespace = conf['data.namespace']
        self.mapping = dict([(poi, idx) for idx, poi in
                             enumerate(self.namespace)]).get
        self.simnum = conf['cf.simnum']
        self.vecs = None
        self.vecsnorm = None

    def train(self, trail_set):
        """ training
        """
        self.vecs = NP.zeros((len(trail_set), len(self.namespace)),
                             dtype=NP.float32)
        for idx, tr in enumerate(trail_set):
            self.vecs[idx, :] = self.process(tr)
        self.vecsnorm = NP.sqrt(NP.sum(self.vecs * self.vecs, axis=1))
        self.logger.info('Data Loaded: {0} ({1}) / {2} MB'.format(
            self.vecs.shape, self.vecs.dtype, self.vecs.nbytes / 1024 / 1024))

    def process(self, tr):
        """ processing tr to vecotr
        """
        vec = NP.zeros(len(self.namespace), dtype=NP.float32)
        for c in tr:
            vec[self.namespace.index(c['poi'])] += 1
        return vec

    def predict(self, trail, tick):  # pylint: disable-msg=W0613
        """ docstring for predict
        """
        vec = self.process(trail)
        est = self.estimates(vec)
        rank = sorted(self.namespace,
                      key=lambda x: est[self.mapping(x)],
                      reverse=True)
        return rank

    def estimates(self, vec):
        """ Estimate the vector
        """
        ents, sims = self.most_similar_to(vec)
        return NP.sum(sims.reshape(-1, 1) * ents, axis=0)

    def most_similar_to(self, vec):
        """ Get top num most similar vectors
        """
        vecnorm = NP.sqrt(NP.sum(vec * vec))
        numerator = NP.sum(vec.reshape(1, -1) * self.vecs, axis=1)
        denominator = vecnorm * self.vecsnorm
        sims = numerator / denominator
        if 0 < self.simnum < sims.shape[0]:
            n_idc = BN.argpartsort(-sims, self.simnum, axis=None)[:self.simnum]
            return NP.array([self.vecs[i] for i in n_idc], dtype=NP.float32), \
                NP.array(sims[n_idc], dtype=NP.float32)
        else:
            return self.vecs, sims


class ColfilterModel(object):
    """ Using cofilter with vectorized trail_set
    """
    def __init__(self, conf):
        """ Constructor

            Arguments:
                conf -- a dict-like object holding configurations
        """
        super(ColfilterModel, self).__init__()
        self.logger = logging.getLogger(
            '%s.%s' % (__name__, type(self).__name__))
        self.namespace = conf['data.namespace']
        self.seglen = conf['cf.segment']
        self.mapping = dict([(poi, idx) for idx, poi in
                             enumerate(self.namespace)]).get

        # Prepare model with kernel specification
        self.vectorizor = KernelVectorizor.from_config(conf)
        self.model = VectorDatabase.from_config(conf)
        self.vecs = None
        self.ck_cnt = None

    def train(self, trail_set):
        """ Train the model
        """
        self.vecs = NP.zeros((len(trail_set), len(self.namespace),
                              self.vectorizor.veclen), dtype=NP.float32)
        self.ck_cnt = NP.zeros((len(trail_set), self.vectorizor.veclen),
                               dtype=NP.byte)
        for idx, tr in enumerate(trail_set):
            self.vecs[idx, :, :] = self.vectorizor.process(tr)
            for c in tr:
                tick = self.vectorizor.get_timeslot(c['tick'])
                self.ck_cnt[idx, tick] += 1 \
                    if self.ck_cnt[idx, tick] < 200 else 0
        self.logger.info('Data Loaded: {0} ({1}) / {2} MB'.format(
            self.vecs.shape, self.vecs.dtype, self.vecs.nbytes / 1024 / 1024))

    def predict(self, tr, tick):
        """ Predicting with colfilter model
        """
        t = self.vectorizor.get_timeslot(tick)
        vec = self.vectorizor.process(tr)[:, t - self.seglen + 1: t + 1]

        self.model.load_data(as_vector_segments(self.vecs, t, self.seglen))
        mask = as_mask(self.ck_cnt, t, self.seglen, level=2)
        est = self.model.estimates_with_mask(vec, t, mask)
        rank = sorted(self.namespace,
                      key=lambda x: est[self.mapping(x), -1],
                      reverse=True)
        return rank


class SparseColfilterModel(object):
    """ Using cofilter with vectorized trail_set
    """
    def __init__(self, conf):
        """ Constructor

            Arguments:
                conf -- a dict-like object holding configurations
        """
        super(SparseColfilterModel, self).__init__()
        self.logger = logging.getLogger(
            '%s.%s' % (__name__, type(self).__name__))
        self.namespace = conf['data.namespace']
        self.seglen = conf['cf.segment']
        self.mapping = dict([(poi, idx)
                             for idx, poi in enumerate(self.namespace)]).get

        # Prepare model with kernel specification
        self.vectorizor = KernelVectorizor.from_config(conf)
        self.model = SparseVectorDatabase.from_config(conf)
        self.vecs = None

    def train(self, trail_set):
        """ Train the model
        """
        beeper = Beeper(self.logger, name='Training', deltacnt=100)
        for _, tr in enumerate(trail_set):
            ck_cnt = NP.zeros(self.vectorizor.veclen, dtype=NP.byte)
            for c in tr:
                tick = self.vectorizor.get_timeslot(c['tick'])
                ck_cnt[tick] += 1 \
                    if ck_cnt[tick] < 200 else 0
            segck = NP.sum(
                as_strided(ck_cnt,
                           shape=(ck_cnt.shape[0] - self.seglen + 1,
                                  self.seglen),
                           strides=(ck_cnt.itemsize, ck_cnt.itemsize)),
                axis=1)
            dsegck = segck[1:].reshape(
                segck.shape[0] / self.seglen,
                self.seglen)
            spvec = self.vectorizor.sp_process(tr)
            spvec.rvecs = as_doublesegments(spvec.rvecs, self.seglen)
            spvec.info = tr[0]['trail_id']
            self.model.extend_dataitems(spvec, dsegck)
            beeper.beep()
        self.logger.info('Sparse Vector indexed {0}'.format(
                         len(self.model.vecs)))
        self.logger.info('Resource usage: {0}MB'.format(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000))

    def predict(self, tr, tick):
        """ Predicting with colfilter model
        """
        t = self.vectorizor.get_timeslot(tick)
        vec = self.vectorizor.process(tr)[:, t - self.seglen + 1: t + 1]

        est = self.model.estimates(vec, t)
        if len(est.shape) != 2:
            self.logger.warn('No similar vectors found for {0}'.format(
                             tr[0]['trail_id']))
            return self.namespace
        rank = sorted(self.namespace,
                      key=lambda x: est[self.mapping(x), -1],
                      reverse=True)
        return rank


def rank_ref(model, history, reftick, refpoi):
    """ running the testing stage
    """
    rank = model.predict(history, reftick)
    return rank.index(refpoi) + 1, refpoi


def iter_test_instance(test_tr_set, segperiod=timedelta(hours=24),
                       filters=None):
    """ generating testing instance
    """
    for trl in test_tr_set:
        trl = sorted(trl, key=lambda x: x['tick'])
        for ref_checkin in trl:
            segend = ref_checkin['tick']
            segstart = segend - segperiod
            segtrail = [c for c in trl if segstart < c['tick'] < segend]
            if len(segtrail) < 2:  # make sure enough check-ins for filter
                continue
            if filters:
                for f in filters:
                    if not f(segtrail):
                        break
                else:
                    if len(segtrail) > 2:  # make sure at least a transition
                        yield segtrail
            else:
                yield segtrail


def print_trail(trail):
    """ printing trail in a compact way
    """
    pois = [c['poi'][-8:-4] for c in trail]
    ticks = [c['tick'].strftime('%H:%M:%S') for c in trail]
    print zip(pois, ticks)


def mkdir_p(dirname):
    """ Try to mkdir
    """
    try:
        os.mkdir(dirname)
    except OSError as err:
        if err.errno == 17:
            if os.path.isdir(dirname):
                return
            raise err


def experiment(conf):  # pylint: disable-msg=R0914
    """ running the experiment
    """
    logger = logging.getLogger('%s.%s' % (__name__, 'experiment'))
    logger.info('--------------------  Experimenting on {0}'
                .format(conf['expr.city.name']))
    logger.info('Reading data from {0}'.format(conf['expr.city.name']))

    data_provider = TextData(conf['expr.city.id'], conf['expr.target'])
    conf['data.namespace'] = data_provider.get_namespace()
    data = data_provider.get_data()
    outdir = os.path.dirname(conf['expr.output'])
    mkdir_p(outdir)
    output = open(conf['expr.output'], 'w')
    model = globals()[conf['expr.model']]
    filters = conf['expr.filters']
    if filters:
        filters = [globals()[f] for f in filters]
    total_trails = checkin_trails(data)
    numfold = conf['expr.folds']
    segperiod = conf['vec.unit'] * conf['cf.segment']

    logger.info('Predicting {0}'.format(conf['expr.target']))
    logger.info('Trails in given dataset: {0}'.format(len(total_trails)))
    for fid, (test_tr, train_tr) in enumerate(folds(total_trails, numfold)):
        if conf['expr.fold_id'] is not None and fid != conf['expr.fold_id']:
            continue

        train_tr_set = list(train_tr)
        test_tr_set = list(test_tr)
        logger.info('----------  Fold: {0}/{1}'
                    .format(fid + 1, conf['expr.folds']))
        logger.info('Checkins: {0} / {1}'
                    .format(sum([len(tr) for tr in test_tr_set]),
                            sum([len(tr) for tr in train_tr_set])))
        logger.info('Trails: {0} / {1}'.
                    format(len(test_tr_set), len(train_tr_set)))

        logger.info('Training...')
        m = model(conf)
        m.train(train_tr_set)

        logger.info('Testing...[Output: {0}]'.format(output.name))
        statcounter = Counter()
        beeper = Beeper(logger, name='Testing', deltacnt=100)
        for segtrl in iter_test_instance(test_tr_set, segperiod, filters):
            htrl = segtrl[:-1]
            reftick = segtrl[-1]['tick']
            refpoi = segtrl[-1]['poi']

            print >> output, rank_ref(m, htrl, reftick, refpoi)
            statcounter.update(['instances'])
            beeper.beep()

        logger.info('Tested trails: {0}'.format(statcounter['instances']))
    output.flush()
    output.close()


if __name__ == '__main__':
    raise Exception('Should run experiment.py instead of this one')
