#!/usr/bin/python
# -*- coding: utf-8 -*-
"""  Stochastic Dynamic Programming library

Implements naive methods of Dynamic Programming (Value Iteration)
to solve *simple* Optimal Stochastic Control problems

classes : SysDescription, DPSolver

Pierre Haessig — November 2013
"""
from __future__ import (division, print_function, unicode_literals,
                        absolute_import)

from .stodynprog import SysDescription, DPSolver


def run_tests(verbose=False):
    '''run the unit tests of stodynprog (using `nose` module)
    
    return True if test run succeeded, False otherwise
    '''
    import nose
    import sys
    print('Stodynprog unit tests:')
    argv = [sys.argv[0]]
    if verbose:
        argv.append('--verbose')
    success = nose.run(argv=argv)
    return success