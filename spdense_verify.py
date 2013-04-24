"""
File: spdense_verify.py
Author: SpaceLis
Email: Wen.Li@tudelft.nl
Github: none
Description:
"""

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
from mlmodels.utils.spindex import from_dense
from mlmodels.model.colfilter import CosineSimilarity, PearsonSimilarity
from mlmodels.model.colfilter import SparseSQRTCosineSimilarity
from datetime import timedelta, datetime
from twhere.trail import KernelVectorizor


EPSILON = 1e-10


def test_cosine():
    """ test cosine similarity
    """
    numpoi = 400
    seglen = 100
    zeros = 100

    a = np.zeros((numpoi, seglen), dtype=np.float32)
    b = np.zeros((numpoi, seglen), dtype=np.float32)
    a[:] = np.random.random((numpoi, seglen))
    b[:] = np.random.random((numpoi, seglen))
    for _ in range(zeros):
        a[np.random.randint(0, numpoi), :] = EPSILON
        b[np.random.randint(0, numpoi), :] = EPSILON
    spa = from_dense(a, None)
    spb = from_dense(b, None)
    assert_equal(a, spa.densify())
    assert_equal(b, spb.densify())
    cs = CosineSimilarity()
    spsqcs = SparseSQRTCosineSimilarity()
    aa = np.array([a], dtype=np.float32)
    refcos = np.dot(a.reshape(-1), b.reshape(-1)) / \
        (np.sqrt(np.dot(a.reshape(-1), a.reshape(-1))) *
         np.sqrt(np.dot(b.reshape(-1), b.reshape(-1))))
    refsqrtcos = np.sqrt(np.dot(a.reshape(-1), b.reshape(-1))) / \
        (np.sqrt(np.dot(a.reshape(-1), a.reshape(-1))) *
            np.sqrt(np.dot(b.reshape(-1), b.reshape(-1))))
    assert_almost_equal(cs.measure_with_mask(aa, b, np.array([True]))[0],
                        refcos, decimal=6)
    assert_almost_equal(cs.sp_measure(spb, [spa])[0], refcos, decimal=6)
    assert_almost_equal(spsqcs.sp_measure(spb, [spa])[0],
                        refsqrtcos, decimal=6)


def test_pearson():
    """ test cosine similarity
    """
    numpoi = 400
    seglen = 100
    zeros = 100

    a = np.zeros((numpoi, seglen), dtype=np.float32)
    b = np.zeros((numpoi, seglen), dtype=np.float32)
    a[:] = np.random.random((numpoi, seglen))
    b[:] = np.random.random((numpoi, seglen))
    for _ in range(zeros):
        a[np.random.randint(0, numpoi), :] = EPSILON
        b[np.random.randint(0, numpoi), :] = EPSILON
    spa = from_dense(a, None)
    spb = from_dense(b, None)
    assert_equal(a, spa.densify())
    assert_equal(b, spb.densify())
    ps = PearsonSimilarity()
    aa = np.array([a], dtype=np.float32)
    refpearson = np.mean((a - np.mean(a)) * (b - np.mean(b))) / \
        (np.std(a) * np.std(b))
    assert_almost_equal(ps.measure_with_mask(aa, b, np.array([True]))[0],
                        refpearson, decimal=6)
    assert_almost_equal(ps.sp_measure(spb, [spa])[0], refpearson, decimal=6)


def test_spvc_vc():
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
    names = dict(globals())
    funcnames = [name for name in names if name.startswith('test')]
    for funcname in funcnames:
        print 'Running...', funcname
        globals()[funcname]()
