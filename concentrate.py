"""
File: concentrate.py
Author: SpaceLis
Email: 0
Github: 0
Description:
"""
import os
import sys


# from
# http://code.activestate.com/recipes/541096-prompt-the-user-for-confirmation/
def confirm(prompt=None, resp=False):
    """prompts for yes or no response from the user. Returns True for yes and
    False for no.

    'resp' should be set to the default value assumed by the caller when
    user simply types ENTER.

    >>> confirm(prompt='Create Directory?', resp=True)
    Create Directory? [y]|n:
    True
    >>> confirm(prompt='Create Directory?', resp=False)
    Create Directory? [n]|y:
    False
    >>> confirm(prompt='Create Directory?', resp=False)
    Create Directory? [n]|y: y
    True

    """

    if prompt is None:
        prompt = 'Confirm'

    if resp:
        prompt = '%s [%s]|%s: ' % (prompt, 'y', 'n')
    else:
        prompt = '%s [%s]|%s: ' % (prompt, 'n', 'y')

    while True:
        ans = raw_input(prompt)
        if not ans:
            return resp
        if ans not in ['y', 'Y', 'n', 'N']:
            print 'please enter y or n.'
            continue
        if ans == 'y' or ans == 'Y':
            return True
        if ans == 'n' or ans == 'N':
            return False


if __name__ == '__main__':
    folddir = sys.argv[1]
    resdir = sys.argv[2]
    if not os.path.isdir(resdir):
        os.mkdir(resdir)
    name = os.listdir(folddir)
    nam = set([n[:-6] for n in name if n.endswith('.res')])
    for n in nam:
        foutname = resdir + '/' + n + '.res'
        if os.path.exists(foutname):
            if not confirm(prompt=foutname + ' exists, overwrite? [y]/n',
                           resp=True):
                continue
        with open(foutname, 'w') as fout:
            skip = False  # Skip those experiments with missing folds
            for i in range(10):
                if not os.path.exists(folddir + '/' + n + '_' + str(i) + '.res'):
                    skip = True
                    print >>sys.stderr, folddir + '/' + n + '_' + str(i) + '.res'
            if skip:
                continue
            for i in range(10):
                with open(folddir + '/' + n + '_' + str(i) + '.res') as fin:
                    for l in fin:
                        fout.write(l)
