#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: anexpr.py
Author: SpaceLis
Changes:
    0.0.1 The first version
Description:
"""
__version__ = '0.0.1'


CONFSTR = """\
source @0/py27/bin/activate && \
(cd @1; python twhere/runner2.py -s \
'{"expr.city.name":"%(city)s","expr.fold_id":%(foldid)d,"expr.output":"%(tmpprefix)s%(output)s","expr.target":"%(target)s"}'; \
hadoop fs -put %(tmpprefix)s%(output)s %(outdir)s; \
rm %(tmpprefix)s%(output)s)\
"""

CITY = ['NY', 'CH', 'LA', 'SF']

def getscript(city='SF', foldid=0, tmpprefix='/tmp/wl-tudelft-', output='test.out', target='base', outdir='test'):
    """ print a configuration
    """
    conf = locals()
    return CONFSTR % conf

def get_cityloop(tmpprefix='/tmp/wl-tudelft-', target='base', outdir='test'):
    """ loop over city
    """
    for c in CITY:
        for f in range(10):
            print getscript(city=c, foldid=f, tmpprefix=tmpprefix, output='%s_%d.res' % (c, f), target=target, outdir=outdir)


def test():
    print getscript(city='NY')

if __name__ == '__main__':
    get_cityloop()
