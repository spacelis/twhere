#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: guianexpr.py
Author: SpaceLis
Email: Wen.Li@tudelft.nl
Github: none
Description: A gui to anexpr.py
"""

import Tkinter as tk
import tkFileDialog as tkfd
from twhere.config import Configuration
import subprocess


class AnexprGui(object):
    """ A gui to anexpr.py
    """
    def __init__(self, master):
        super(AnexprGui, self).__init__()
        self.frame = tk.Frame(master)
        self.frame.pack()

        #TODO use eval for correct setting types
        # Save button
        btnSave = tk.Button(self.frame, text='Save')
        btnSave.grid(row=0, column=0, columnspan=2, sticky=tk.E)
        btnSave.bind('<Button-1>', lambda e: self.save())

        self.doexpansion = tk.IntVar(self.frame)
        self.doexpansion.set(1)
        ckExpand = tk.Checkbutton(self.frame, text='Expand', onvalue=1,
                                  offvalue=0, variable=self.doexpansion)
        ckExpand.grid(row=1, column=0, columnspan=2, sticky=tk.E)

        self.scripttype = tk.IntVar(self.frame)
        self.scripttype.set(1)
        rbHadoop = tk.Radiobutton(self.frame, text='Hadoop',
                                  variable=self.scripttype, val=0)
        rbHadoop.grid(row=2, column=0, columnspan=2, sticky=tk.E)
        rbMultiproc = tk.Radiobutton(self.frame, text='Multiproc',
                                     variable=self.scripttype, val=1)
        rbMultiproc.grid(row=3, column=0, columnspan=2, sticky=tk.E)

        self.scriptconf = dict()
        self.addEntry('NAME', '', self.scriptconf, 4, True)
        self.addEntry('OUTDIR', '', self.scriptconf, 5, True)
        self.addEntry('HOUTDIR', '', self.scriptconf, 6, True)

        self.conf = Configuration()
        self.confvar = dict()
        for idx, (k, v) in enumerate(sorted(self.conf,
                                            key=lambda (x, y): x), 7):
            self.addEntry(k, str(v), self.confvar, idx)
        self.freqChanged()

    def freqChanged(self):
        """ Change some settings for freqent use
        """
        self.confvar['expr.city.name']['var'].set('["NY","CH","SF","LA"]')
        self.updateColor('expr.city.name')

    def addEntry(self, name, defval, vargroup, row, compulsory=False):
        """ Add an entry in UI
        """
        lbl = tk.Label(self.frame, text=name,
                       fg='red' if compulsory else 'black')
        lbl.grid(row=row, column=0, sticky=tk.W)
        lbl.bind('<Button-1>', lambda e: self.resetValue(name))
        txtvar = tk.StringVar()
        txtvar.set(defval)
        txt = tk.Entry(self.frame, textvariable=txtvar, width=50,
                       fg='black' if compulsory else 'grey')
        txt.grid(row=row, column=1, sticky=tk.W)
        txt.bind('<FocusOut>', lambda e: self.focusOut(name))
        vargroup[name] = {'var': txtvar,
                          'defval': defval,
                          'label': lbl,
                          'entry': txt}

    def resetValue(self, name):
        """ reset the asscoiate var
        """
        if name not in self.confvar:
            return
        self.confvar[name]['var'].set(self.confvar[name]['defval'])
        self.updateColor(name)
        print 'reset'

    def focusOut(self, name):
        """ mark given confitem that is manually input
        """
        if name not in self.confvar:
            return
        var = self.confvar[name]['var']
        var.set(var.get().strip())
        if len(var.get()) == 0:
            self.resetValue(name)
        else:
            self.updateColor(name)

    def updateColor(self, name):
        """ Update the color on widgets
        """
        if name not in self.confvar:
            return
        txt = self.confvar[name]['entry']
        if not self.ischanged(name):
            txt['fg'] = 'grey'
        else:
            txt['fg'] = 'black'

    def ischanged(self, name):
        """ Return True of the value is changed for the confitem
        """
        if name not in self.confvar:
            return False
        val = self.confvar[name]['var'].get()
        defval = self.confvar[name]['defval']
        return not (len(val) == 0 or val == defval)

    def save(self):
        """ Save the script
        """
        conf = dict()
        for name in self.confvar.iterkeys():
            if self.ischanged(name):
                conf[name] = self.confvar[name]['var'].get()
        confstr = "{" + ', '.join(['"%s":%s' % (k, v)
                                   for k, v in conf.iteritems()]) + "}"
        cmd = ['python', 'twhere/anexpr.py']
        cmd += [] if not self.doexpansion.get() else ['-e']
        cmd += ['-i', confstr]
        cmd += ['-b', self.scriptconf['HOUTDIR']['var'].get()] \
            if self.scripttype.get() == 0 else []
        cmd += ['-o', self.scriptconf['OUTDIR']['var'].get()]
        cmd += [self.scriptconf['NAME']['var'].get()]
        print ' '.join(cmd)
        with tkfd.asksaveasfile(mode='w', defaultextension='pconf') as fout:
            fout.write(subprocess.check_output(cmd))

if __name__ == '__main__':
    root = tk.Tk()
    app = AnexprGui(root)
    root.mainloop()
