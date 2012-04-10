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
site.addsitedir('.')
import logging
import sys

import MySQLdb as sql
import numpy as np
from model.colfilter import MemoryCFModel, cos_NB, lincombine, batchsim

from trail import TrailGen

conn = sql.connect(user='root',
        read_default_file='/home/wenli/devel/mysql/my.cnf',
        db='geotweet')


class CategoriesX24h(object):
    """ a collection of method for parsing trails into items
    """
    def __init__(self, categories):
        self.namespace = [i for i in categories]

    def size(self):
        return len(self.namespace), 24

    def parse_trail(self, trail):
        """ Parse a trail into a matrix
        """
        tmp = np.zeros((len(self.namespace), 24), dtype=np.float32)
        tmp.fill(np.float32('nan'))
        for c in trail:
            i, j = self.columnfunc(c)
            tmp[i, j] = 1
        return tmp.flatten()

    @staticmethod
    def scale(tick):
        return str(tick.hour)

    def columnfunc(self, checkin):
        return self.namespace.index(checkin['poi']), CategoriesX24h.scale(checkin['tick'])


class DiscreteTxC(object):
    """ A trail is treated as a set of (Place, Time) items representing a staying at a place
        and the time dimension is discrete as hours.
    """
    def __init__(self, parser):
        super(DiscreteTxC, self).__init__()
        self.parser = parser
        self.data = None
        self.model = None

    def train(self, trails):
        """ Parse trails in to arrays so that they can be used in CF
        """
        tr_iter = iter(trails)
        self.data = np.array([self.parser.parse_trail(tr_iter.next())])
        for tr in tr_iter:
            self.data = np.append(self.data, [self.parser.parse_trail(tr)], 0)
        self.model = MemoryCFModel(self.data, lambda ent, ents: batchsim(ent, ents, cos_NB), lincombine)
        logging.info('Train finished!')

    def predict(self, trail, time):
        """ predict a trail of stay at `time`
        """
        ent = self.parser.parse_trail(trail)
        _ent = self.model.estimate(ent)
        p = _ent.reshape(self.parser.size())[:, time]
        return [self.parser.namespace[i] for i in np.argsort(p)][::-1]

    def evaluate(self, trails):
        """ evaluate the model using a set of `trails`
        """
        result = np.zeros(len(trails), dtype=np.int32)
        for idx, tr in enumerate(trails):
            ref, t = self.parser.columnfunc(tr[-1])
            prefix = tr[:-1]
            rank = self.predict(prefix, t)
            result[idx] = rank.index(self.parser.namespace[ref])
        return result


def cv_folds(fold, table, city):
    cur = conn.cursor(cursorclass=sql.cursors.DictCursor)
    total = cur.execute('''select
                                c.tr_id as trail_id,
                                p.category as poi,
                                c.created_at as tick,
                                c.text as text
                            from %s as c
                                left join checkins_place as p
                                on c.pid = p.id
                            where c.isbot is null and p.city='%s'
                            order by c.uid, c.created_at''' %
                            (table, city))
    trainset = list()
    testset = list()
    logging.info('Total Checkins: %d' % (total,))
    rowset = list(cur)

    for i in range(fold):
        for cnt, row in enumerate(rowset):
            if cnt > float(total) * i / fold and cnt < float(total) * (i + 1) / fold:
                testset.append(row)
            else:
                trainset.append(row)
        test_poi = set(c['poi'] for c in testset)
        shared_poi = set([c['poi'] for c in trainset]) & test_poi
        logging.info('POIs shared / POIs tested: %d / %d' % (len(shared_poi), len(test_poi)))
        yield trainset, testset


def experiment():
    """ running the experiment
    """
    cur = conn.cursor(cursorclass=sql.cursors.DictCursor)
    cur.execute('select distinct id from category')
    parser = CategoriesX24h([c['id'] for c in cur])
    logging.info('Reading data...')
    for trainset, testset in cv_folds(10, 'checkins_6', sys.argv[1]):
        m = DiscreteTxC(parser)
        logging.info('Training...')
        m.train([tr for tr in TrailGen(trainset, lambda x:x['trail_id'])])
        logging.info('Testing...')
        for e in m.evaluate([tr for tr in TrailGen(testset, lambda x:x['trail_id'])]):
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

    parser = CategoriesX24h(['ewi', 'home', 'canteen'])
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
