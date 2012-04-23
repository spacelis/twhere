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


class MySQLData(object):
    """ Data provide from MySQL
    """
    def __init__(self, city, poicol):
        super(MySQLData, self).__init__()
        self.city, self.poicol = city, poicol
        self.conn = sql.connect(user='root',
            read_default_file='/home/wenli/devel/mysql/my.cnf',
            db='geotweet')

    def get_namespace(self):
        """ Return the list of unique labels in poicol
        """
        cur = self.conn.cursor(cursorclass=sql.cursors.DictCursor)
        cur.execute('select distinct id as id from category')
        ns = [c['id'] for c in cur]
        return ns

    def get_data(self):
        """ Return a list of checkins
        """
        cur = self.conn.cursor(cursorclass=sql.cursors.DictCursor)
        total = cur.execute('''select
                                    c.tr_id as trail_id,
                                    p.%s as poi,
                                    c.created_at as tick
                                from %s as c
                                    left join checkins_place as p
                                    on c.pid = p.id
                                where c.isbot is null and p.city='%s'
                                order by c.uid, c.created_at''' %
                                (self.poicol, 'checkins_6', self.city))
        logging.info('Total Checkins: %d' % (total,))
        return list(cur)
