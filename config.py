
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
                  'cf.similarity': 'SparseCosineSimilarity',
                  'cf.aggregator': 'LinearCombination',
                  'expr.city.name': 'NY',
                  'expr.city.id': '27485069891a7938',
                  'expr.target': 'category',
                  'expr.model': 'ColfilterModel',
                  'expr.output': None,                  # Must be override
                  'expr.folds': 10,
                  'expr.fold_id': None,                 # None for running all folds
                  'data.namespace': None,               # will be override during data loading phase
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
            fout.write(json.dumps(Configuration.flatten(self.conf), encoding='utf-8'))
        return fname

    def load(self, fname):
        """ Load current configuration with fname
        """
        with open(fname) as fin:
            self.conf = Configuration.deflatten(json.loads(fin.read(), encoding='utf-8'))

    def update(self, delta):
        """ Update the configuration by a delta configuration
        """
        self.conf.update(delta)

    def __getitem__(self, key):
        return self.conf[key]

    def __setitem__(self, key, val):
        self.conf[key] = val

    def __delitem__(self, key):
        del self[key]

    def __len__(self):
        return len(self.conf)

if __name__ == '__main__':
    raise Exception('Should run experiment.py')
