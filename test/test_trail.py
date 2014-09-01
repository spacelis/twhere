#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: test_trail.py
Author: SpaceLis
Changes:
    0.0.1 The first version
Description:
"""
__version__ = '0.0.1'


import unittest
from datetime import datetime, timedelta
from ..trail import checkin_trails
from ..trail import BinaryVectorizor
from ..trail import KernelVectorizor
from ..trail import TrailVectorSegmentSet
from ..trail import as_vector_segments
from ..trail import future_diminisher
from ..trail import gaussian_pdf
from ..trail import kernel_smooth
import numpy as NP
from numpy.testing import assert_equal
from matplotlib import pyplot as PLT
#from collections import Counter
from itertools import izip_longest
import unittest


AXIS = NP.linspace(0., 24 * 3600., num=100, endpoint=False)

class TestTrail(unittest.TestCase):

    """Test case docstring."""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testIterTrails(self):
        """ Test itertrails
        """
        testinput = [[1, 2],
                    [1, 4],
                    [2, 3],
                    [2, 3],
                    [2, 7],
                    [3, 7],
                    [3, 7],
                    [3, 3]]
        testoutput = [[[1, 2], [1, 4]],
                    [[2, 3], [2, 3], [2, 7]],
                    [[3, 7], [3, 7], [3, 3]]]
        cktr = checkin_trails(testinput, lambda x: x[0])
        for x, y in izip_longest(cktr, testoutput):
            self.assertEqual(x, y)
            self.assertEqual(list(x[1]), list(y[1]))


    def testBinaryVectorizor(self):
        """ test
        """
        bvz = BinaryVectorizor(['a', 'b', 'c'],
                            unit=timedelta(seconds=3600),
                            epoch=datetime(2010, 12, 13),
                            eschatos=datetime(2010, 12, 14),
                            accumulated=False)
        data = ['a 2010-12-13 04:07:12',
                'b 2010-12-13 04:20:39',
                'c 2010-12-13 06:09:51']
        checkin_set = [{"poi": x,
                        "tick": datetime.strptime(y, '%Y-%m-%d %H:%M:%S')}
                    for x, y in [s.split(' ', 1) for s in data]]
        assert_equal(bvz.process(checkin_set).T,
                    [[0.,  0.,  0.,  0.,  1.,  0.,
                    0.,  0.,  0.,  0.,  0.,  0.,
                    0.,  0.,  0.,  0.,  0.,  0.,
                    0.,  0.,  0.,  0.,  0.,  0.],
                    [0.,  0.,  0.,  0.,  1.,  0.,
                    0.,  0.,  0.,  0.,  0.,  0.,
                    0.,  0.,  0.,  0.,  0.,  0.,
                    0.,  0.,  0.,  0.,  0.,  0.],
                    [0.,  0.,  0.,  0.,  0.,  0.,
                    1.,  0.,  0.,  0.,  0.,  0.,
                    0.,  0.,  0.,  0.,  0.,  0.,
                    0.,  0.,  0.,  0.,  0.,  0.]])


    def testKernelVectorizor(self):
        """ test
        """
        kvz = KernelVectorizor(['a', 'b', 'c'],
                            unit=timedelta(seconds=3600),
                            epoch=datetime(2010, 12, 13),
                            eschatos=datetime(2010, 12, 14),
                            accumulated=False)
        data = ['a 2010-12-13 04:07:12',
                'b 2010-12-13 04:20:39',
                'b 2010-12-13 06:09:51']
        checkin_set = [{'poi': x,
                        'tick': datetime.strptime(y, '%Y-%m-%d %H:%M:%S')}
                    for x, y in [s.split(' ', 1) for s in data]]
        v = kvz.process(checkin_set)
        PLT.plot(range(24), v[0, :])
        PLT.plot(range(24), v[1, :])
        PLT.plot(range(24), v[2, :])
        PLT.show()
        # should see two peaks for one line and one peak for another line
        assert_equal(kvz.get_timeslot(datetime.strptime('2010-12-13 23:17:23',
                                                        '%Y-%m-%d %H:%M:%S')), 23)


    def testTrailSegments(self):
        """ test
        """
        bvz = BinaryVectorizor(['a', 'b', 'c'],
                            unit=timedelta(seconds=3600),
                            epoch=datetime(2010, 12, 13),
                            eschatos=datetime(2010, 12, 14),
                            accumulated=False)
        data = ['1 a 2010-12-13 04:07:12',
                '1 b 2010-12-13 05:20:39',
                '2 c 2010-12-13 04:09:51',
                '2 c 2010-12-13 05:09:51']
        checkin_set = [{'trail_id': x,
                        'poi': y,
                        'tick': datetime.strptime(z, '%Y-%m-%d %H:%M:%S')}
                    for x, y, z in [s.split(' ', 2) for s in data]]
        vecs = NP.zeros((bvz.veclen, 2, 3), dtype=NP.float64)
        for idx, trail in enumerate(checkin_trails(checkin_set)):
            bvz.process(trail, vecs[:, idx, :])
        ts = TrailVectorSegmentSet(vecs, 2)
        tsl = list(ts.enumerate_segment_sets(
            bvz.get_timeslot(datetime(2010, 12, 13, 15, 20))))
        assert_equal(tsl[2][1], NP.array([[[1., 0., 0.], [0., 0., 1.]],
                                        [[0., 1., 0.], [0., 0., 1.]]]))


    def testAsVectorSegments(self):
        """ test
        """
        vecs = NP.zeros((2, 3, 15), dtype=int)
        for i in range(vecs.shape[2]):
            vecs[:, :, i] = i
        v = as_vector_segments(vecs, 13, 4)
        # test correctness of segmentation
        assert_equal(NP.max(v[0], axis=1), [[2, 3, 4, 5],
                                            [6, 7, 8, 9],
                                            [10, 11, 12, 13]])


    def testSegmentationReference(self):
        """ test
        """
        vecs = NP.zeros((2, 3, 15), dtype=int)
        for i in range(vecs.shape[2]):
            vecs[:, :, i] = i
        v = as_vector_segments(vecs, 13, 4)

        # Should be flattened
        vf = v.reshape((v.shape[0], v.shape[1], -1))

        # Should not be a reference
        # TODO check whether this need to be a reference
        vf[0, 1, 5] = 20
        self.assertNotEqual(vecs[1, 1, 7], 20)


class TestTrailFilters(unittest.TestCase):

    """Trails.*filter"""

    def setUp(self):
        dt = lambda s: datetime.strptime("2013-01-01 " + s, "%Y-%m-%d %H:%M")
        mktrails = lambda seq: [[{'tick': dt(tick)} for tick in s] for s in seq]
        self.trails = mktrails([
            [ '08:30', '09:30', '22:20', '22:30' ],
            [ '08:30', '21:30', '22:00', '23:00' ],
            [ '08:30', '22:30', '22:31', '22:32' ],
            [ '22:29', '22:30', '22:31', '22:32' ],
        ])

    def tearDown(self):
        pass

    def test_future_diminisher(self):
        for t, l in zip(self.trails, [3, 4, 2, 1]):
            self.assertEqual(len(future_diminisher(t, 1)), l)
        for t, l in zip(self.trails, [3, 2, 2, 1]):
            self.assertEqual(len(future_diminisher(t, 2)), l)


class TestKDE(unittest.TestCase):

    """ Testing KDE functions """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_gaussian_pdf(self):
        """ Gaussian pdf
        """
        pdf = gaussian_pdf(AXIS, 6 * 3600., (7200.,))
        PLT.plot(AXIS, pdf)
        PLT.show()

    def test_kernel_smooth(self):
        """ kernel_smooth
        """
        pdf = kernel_smooth(AXIS, [6 * 3600., 12 * 3600., 18 * 3600.], (3600., ))
        PLT.plot(AXIS, pdf)
        PLT.show()

    def test_spvc_vc(self):
        """ Test the consistency of sparse vectorizor and normal vectorizor
        """
        kvz = KernelVectorizor(['a', 'b', 'c'],
                            unit=timedelta(seconds=3600),
                            epoch=datetime(2010, 12, 13),
                            eschatos=datetime(2010, 12, 14),
                            accumulated=False)
        data = ['a 2010-12-13 04:07:12',
                'b 2010-12-13 04:20:39',
                'b 2010-12-13 06:09:51']
        checkin_set = [{'poi': x,
                        'tick': datetime.strptime(y, '%Y-%m-%d %H:%M:%S')}
                    for x, y in [s.split(' ', 1) for s in data]]
        spa = kvz.sp_process(checkin_set)
        a = spa.densify()
        b = kvz.process(checkin_set)
        assert_equal(a, b)

if __name__ == '__main__':
    pass
