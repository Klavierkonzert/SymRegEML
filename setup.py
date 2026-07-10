#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 10:31:13 2021
@author: ying

Modified: 2026
@author: Klavierkonzert
"""


from setuptools import setup

setup(name='bsr',
      version='0.1',
      description='A Bayesian MCMC based Symbolic Regression Algorithm on EML-trees',
      author='Klavierkonzert',
      author_email='77981618+Klavierkonzert@users.noreply.github.com',
      url='https://github.com/Klavierkonzert/SymRegEML',
      py_modules = ['bsr.bsr_class','bsr.BSR','bsr.funcs'],
      package_dir = {'bsr':'codes'}
)
