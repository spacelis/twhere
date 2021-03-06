#!python
# -*- coding: utf-8 -*-
"""
File: dataprov.py
Author: SpaceLis
Changes:
    0.0.1 The first version
Description:
    Data provider for the experiments
"""
__version__ = '0.0.1'


import re
import logging
from datetime import datetime, date, timedelta
import numpy as NP


class DataProvider(object):
    """ Providing data
    """
    def __init__(self):
        super(DataProvider, self).__init__()
        self.data = None
        self.namespace = None
        self.logger = logging.getLogger(
            '%s.%s' % (__name__, type(self).__name__))

    def get_namespace(self):
        """ Return the list of unique labels in poicol
        """
        return self.namespace

    def get_data(self):
        """ Return a list of checkins
        """
        return self.data


class MySQLData(DataProvider):
    """ Data provide from MySQL
    """
    def __init__(self, city, poicol):
        super(MySQLData, self).__init__()
        import MySQLdb as sql
        self.logger.info('Using MySQL data provider')
        self.city, self.poicol = city, poicol
        if self.poicol == 'base':
            idname = 'base'
            colname = 'base_category'
        elif self.poicol == 'category':
            idname = 'id'
            colname = 'category'
        self.conn = sql.connect(
            user='root',
            read_default_file='/home/wenli/devel/mysql/my.cnf',
            db='geotweet')
        cur = self.conn.cursor(cursorclass=sql.cursors.DictCursor)
        cur.execute('select distinct %s as id from category' % (idname,))
        self.namespace = [c['id'] for c in cur]

        total = cur.execute('''select
                                    c.tr_id as trail_id,
                                    p.%s as poi,
                                    c.created_at as tick
                                    c.pid as pid
                                from %s as c
                                    left join checkins_place as p
                                    on c.pid = p.id
                                where c.isbot is null and p.city='%s'
                                order by c.uid, c.created_at''' %
                            (colname, 'checkins_6', self.city))
        self.logger.info('Total Checkins: %d' % (total, ))
        self.data = list(cur)


class TextData(DataProvider):
    """ Data provider from text files
    """
    SEPARATOR = re.compile(r'\t')

    def __init__(self, datafile, nsfile):
        super(TextData, self).__init__()
        datapath = 'data/%s_%s_data.table' % (datafile, nsfile)
        nspath = 'data/%s_ns.table' % (nsfile,)
        self.logger.info('Text data provider: ' + datapath + ', ' + nspath)
        self.data = list()
        self.namespace = list()
        with open(datapath) as fin:
            for line in fin:
                tr_id, poi, tick, pid = TextData.SEPARATOR.split(line.strip())
                tmp = {'trail_id': tr_id,
                       'poi': poi,
                       'tick': datetime.strptime(tick, '%Y-%m-%d %H:%M:%S'),
                       'pid': pid}
                self.data.append(tmp)
        with open(nspath) as fin:
            for line in fin:
                self.namespace.append(line.strip())


class RandomData(DataProvider):
    """ Data provider from text files
    """
    def __init__(self, nsfile, length=3, number=3000):
        super(RandomData, self).__init__()
        nspath = 'data/%s_ns.table' % (nsfile,)
        self.logger.info('Random data provider: length=%d, %s' %
                         (length, nspath))
        self.namespace = list()
        with open(nspath) as fin:
            for line in fin:
                self.namespace.append(line.strip())

        self.data = list()
        lengths = NP.ceil(NP.random.standard_exponential(number))
        print lengths
        for idx, length in enumerate(lengths):
            for _ in range(int(length)):
                tr_id = str(idx)
                poi = self.namespace[NP.random.randint(len(self.namespace))]
                tick = str(datetime.fromordinal(date.today().toordinal()) +
                           timedelta(seconds=NP.random.randint(3600 * 24)))
                pid = str(NP.random.randint(10000))
                tmp = {'trail_id': tr_id,
                       'poi': poi,
                       'tick': datetime.strptime(tick, '%Y-%m-%d %H:%M:%S'),
                       'pid': pid}
                self.data.append(tmp)
