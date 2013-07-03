#!/usr/bin/python
# -*- coding: utf-8 -*-
""" test for the Stochastic Dynamic Programming library (StoDynProg)

Pierre Haessig — July 2013
"""

from __future__ import division, print_function, unicode_literals

from nose.tools import assert_equal

import sys
try:
    import stodynprog
except ImportError:
    sys.path.append('..')
    import stodynprog


def test_zero_cost():
    'dummy test of the "zero cost" function that returns zero whatever the input'
    assert_equal(stodynprog._zero_cost(), 0.)
    assert_equal(stodynprog._zero_cost(1), 0.)
    assert_equal(stodynprog._zero_cost(1,2), 0.)
    assert_equal(stodynprog._zero_cost(1,2,3), 0.)

def test_enforce_sig_len():
    pass
