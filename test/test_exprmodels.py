#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: test_twhere.py
Author: SpaceLis
Changes:
    0.0.1 The first version
Description:
"""
__version__ = '0.0.1'

from datetime import datetime
from numpy.testing import assert_equal
from ..config import Configuration
from ..trail import checkin_trails
from ..exprmodels import ColfilterModel
from ..exprmodels import rank_ref
from ..exprmodels import PredictingTimeMajority
import unittest

import logging
logging.basicConfig(format='%(asctime)s %(name)s [%(levelname)s] %(message)s',
                    level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class TestModels(unittest.TestCase):

    """ Test models """

    def setUp(self):
        self.column = ['trail_id', 'poi', 'tick', 'text']
        self.train = [['1', 'home', datetime(2011, 1, 1, 6, 30), 'test'],
                      ['1', 'ewi', datetime(2011, 1, 1, 9, 30), 'test'],
                      ['1', 'canteen', datetime(2011, 1, 1, 12, 30), 'test'],
                      ['1', 'ewi', datetime(2011, 1, 1, 14, 30), 'test'],

                      ['2', 'ewi', datetime(2011, 1, 1, 11, 30), 'test'],
                      ['2', 'canteen', datetime(2011, 1, 1, 12, 30), 'test'],
                      ['2', 'ewi', datetime(2011, 1, 1, 14, 30), 'test'],
                      ['2', 'home', datetime(2011, 1, 1, 18, 30), 'test']]

        self.trainset = [dict([(key, val)
                               for key, val in zip(self.column, checkin)])
                         for checkin in self.train]

        self.trail = [['3', 'home', datetime(2011, 1, 1, 6, 30), 'test'],
                      ['3', 'ewi', datetime(2011, 1, 1, 9, 30), 'test'],
                      ['3', 'canteen', datetime(2011, 1, 1, 12), 'test'],
                      ['3', 'ewi', datetime(2011, 1, 1, 14, 25), 'test']]

        self.trailset = [dict([(key, val)
                               for key, val in zip(self.column, checkin)])
                         for checkin in self.trail]

    def tearDown(self):
        pass

    def test_colfiltermodel(self):
        """ Test the model
        """
        conf = Configuration()
        conf['data.namespace'] = ['home', 'ewi', 'canteen']
        conf['vec.kernel'] = 'uniform_pdf'

        m = ColfilterModel(conf)
        LOGGER.info('Training...')
        m.train(checkin_trails(self.trainset, lambda x: x['trail_id']))
        LOGGER.info('Testing...')
        for tr in checkin_trails(self.trailset, lambda x: x['trail_id']):
            e = rank_ref(m, tr[:-1], tr[-1]['tick'], tr[-1]['poi'])
            assert_equal(e, 1)

    def test_PTMmodel(self):
        """ Test the model
        """
        conf = Configuration()
        conf['data.namespace'] = ['home', 'ewi', 'canteen']
        conf['vec.unit'] = '1h'
        conf['vec.epoch'] = datetime(2011, 1, 1)
        conf['vec.eschatos'] = datetime(2011, 2, 1)

        m = PredictingTimeMajority(conf)
        LOGGER.info('Training...')
        m.train(checkin_trails(self.trainset, lambda x: x['trail_id']))
        LOGGER.info('Testing...')
        for tr in checkin_trails(self.trailset, lambda x: x['trail_id']):
            e = rank_ref(m, tr[:-1], tr[-1]['tick'], tr[-1]['poi'])
            assert_equal(e, 1)


if __name__ == '__main__':
    pass
