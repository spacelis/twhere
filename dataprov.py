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


import MySQLdb as sql
import logging
from datetime import datetime


class MySQLData(object):
    """ Data provide from MySQL
    """
    def __init__(self, city, poicol):
        super(MySQLData, self).__init__()
        logging.info('Using MySQL data provider')
        self.city, self.poicol = city, poicol
        if self.poicol == 'base':
            idname = 'base'
            colname = 'base_category'
        elif self.poicol == 'category':
            idname = 'id'
            colname = 'category'
        self.conn = sql.connect(user='root',
            read_default_file='/home/wenli/devel/mysql/my.cnf',
            db='geotweet')
        cur = self.conn.cursor(cursorclass=sql.cursors.DictCursor)
        cur.execute('select distinct %s as id from category' % (idname,))
        self.namespace = [c['id'] for c in cur]

        total = cur.execute('''select
                                    c.tr_id as trail_id,
                                    p.%s as poi,
                                    c.created_at as tick
                                from %s as c
                                    left join checkins_place as p
                                    on c.pid = p.id
                                where c.isbot is null and p.city='%s'
                                order by c.uid, c.created_at''' %
                                (colname, 'checkins_6', self.city))
        logging.info('Total Checkins: %d' % (total,))
        self.data = list(cur)

    def get_namespace(self):
        """ Return the list of unique labels in poicol
        """
        return self.namespace

    def get_data(self):
        """ Return a list of checkins
        """
        return self.data


class TextData(object):
    """ Data provider from text files
    """
    def __init__(self, datafile, nsfile):
        super(TextData, self).__init__()
        logging.info('Using text data provider')
        self.data = list()
        self.namespace = list()
        with open('data/%s_%s_data.table' % (datafile, nsfile)) as fin:
            for line in fin:
                tr_id, poi, tick = line.strip().split('\t')
                tmp = {'trail_id': tr_id, 'poi': poi, 'tick': datetime.strptime(tick, '%Y-%m-%d %H:%M:%S')}
                self.data.append(tmp)
        with open('data/%s_ns.table' % (nsfile,)) as fin:
            for line in fin:
                self.namespace.append(line.strip())

    def get_namespace(self):
        """ Return a list of unique labels in the predicting field
        """
        return self.namespace

    def get_data(self):
        """ Return a list of checkins
        """
        return self.data
