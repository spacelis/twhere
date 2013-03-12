
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: config.py
Author: SpaceLis
Changes:
    0.0.2 dict like configuration to ease configuration passing on
    0.0.1 The first version
Description:
    This is an example of config.py which will be used in twhere.py
"""
__version__ = '0.0.2'


import uuid
import json
import datetime

DEFAULT_CONFIG = {'vec.unit': datetime.timedelta(seconds=24 * 36),
                  'vec.epoch': datetime.datetime(2010, 6, 1),
                  'vec.eschatos': datetime.datetime(2011, 6, 1),
                  'vec.timeparser': None,
                  'vec.kernel': 'gaussian_pdf',
                  'vec.kernel.params': (3600.,),
                  'vec.kernel.accumulated': True,
                  'vec.kernel.normalized': False,
                  'vec.kernel.reuse_orig': False,
                  'discounting.func': 'exponent',
                  'discounting.params': {'l': 1 / 3600.},
                  'cf.segment': 100,
                  'cf.simnum': 20,
                  'cf.similarity': 'CosineSimilarity',
                  'cf.aggregator': 'LinearCombination',
                  'expr.city.name': 'NY',
                  'expr.city.id': '27485069891a7938',
                  'expr.target': 'category',
                  'expr.model': 'ColfilterModel',
                  'expr.output': None,
                  'data.namespace': None,
                  }


class Configuration(object):
    """ An object holding all configuration
    """
    def __init__(self, **kargs):
        super(Configuration, self).__init__()
        self.conf = dict(DEFAULT_CONFIG)
        self.conf.update(kargs)

    @staticmethod
    def flatten(conf):
        """ convert the non-interpretables into strings
        """
        flatobj = dict(conf)
        flatobj['vec.unit'] = repr(conf['vec.unit'])
        flatobj['vec.epoch'] = repr(conf['vec.epoch'])
        flatobj['vec.eschatos'] = repr(conf['vec.eschatos'])
        return flatobj

    @staticmethod
    def deflatten(flatobj):
        """ convert the strings into objects
        """
        conf = dict(flatobj)
        conf['vec.unit'] = eval(flatobj['vec.unit'])
        conf['vec.epoch'] = eval(flatobj['vec.epoch'])
        conf['vec.eschatos'] = eval(flatobj['vec.eschatos'])
        return conf

    def save(self, fname=None):
        """ Save current configuration
        """
        if fname is None:
            fname = str(uuid.uuid1()) + '.tempconf'
        with open(fname, 'w') as fout:
            json.dump(Configuration.flatten(self.conf), fout)
        return fname

    def load(self, fname):
        """ Load current configuration with fname
        """
        with open(fname) as fin:
            self.conf = Configuration.deflatten(json.load(fin))

    def __getitem__(self, key):
        """ Return the property value
        """
        return self.conf[key]

    def __setitem__(self, key, val):
        """ Set the property of key by val
        """
        self.conf[key] = val


if __name__ == '__main__':
    raise Exception('Should run experiment.py')
