#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: beeper.py
Author: SpaceLis
Changes:
    0.0.1 The first version
Description:
"""
__version__ = '0.0.1'


from time import time

class Beeper(object):
    """ A beeper asserting alive
    """
    def __init__(self, logger, name='aBeeper', deltacnt=10, deltatime=300, showmsg=None):
        super(Beeper, self).__init__()
        self.logger = logger
        self.name = name
        self.deltacnt = deltacnt
        self.deltatime = deltatime
        self.showmsg = showmsg

        self.previous = time()
        self.cnt = self.deltacnt
        self.total = 0
        self.logger.info('Beeper %s start!' % self.name)

    def beep(self):
        """ Log an INFO level message."""
        if ((self.previous is not None and time() > self.previous + self.deltatime)
                or (self.deltacnt is not None and self.cnt <= 0)):
            msg = ''
            if self.showmsg:
                msg = ' [%s] ' % self.showmsg()
            self.logger.info('%s -- BEEP!%s#%d' % (self.name, msg, self.total))
            if self.previous is not None:
                self.previous = time()
            if self.deltacnt is not None:
                self.cnt = self.deltacnt
        if self.deltacnt is not None:
            self.cnt -= 1
        self.total += 1
