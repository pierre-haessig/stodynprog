#!/usr/bin/python
# -*- coding: utf-8 -*-
""" test for the Stochastic Dynamic Programming library (StoDynProg)

Pierre Haessig — July 2013
"""

from __future__ import division, print_function, unicode_literals

from nose.tools import assert_equal, assert_true, assert_raises

import stodynprog
from stodynprog.stodynprog import _zero_cost, _enforce_sig_len

def test_zero_cost():
    'test of the "zero cost" function'
    assert_equal(_zero_cost(), 0.)
    assert_equal(_zero_cost(1), 0.)
    assert_equal(_zero_cost(1,2), 0.)
    assert_equal(_zero_cost(1,2,3), 0.)

def test_enforce_sig_len():
    "function's  signature length enforcement"
    # shorthand name:
    enforce_sig_len = _enforce_sig_len
    # Create some functions to check
    def f0():
        pass
    def f1(x):
        pass
    def f2(x, y):
        pass
    
    # Potential signatures
    arg0 = []
    arg1 = ['x']
    arg2 = ['x','y']
    
    # f0 should accept 0 argument:
    assert_true(enforce_sig_len(f0, arg0))
    assert_raises(ValueError, enforce_sig_len, f0, arg1)
    assert_raises(ValueError, enforce_sig_len, f0, arg2)
    # f1 should accept 1 argument:
    assert_raises(ValueError, enforce_sig_len, f1, arg0)
    assert_true(enforce_sig_len(f1, arg1))
    assert_raises(ValueError, enforce_sig_len, f1, arg2)
    # f2 should accept 2 arguments:
    assert_raises(ValueError, enforce_sig_len, f1, arg0)
    assert_raises(ValueError, enforce_sig_len, f2, arg1)
    assert_true(enforce_sig_len(f2, arg2))
    # Finally, check the error message
    try:
        enforce_sig_len(f1, arg2)
    except ValueError as e:
        pass
    assert_equal(e.message, "'f1' should accept 2 args (x, y), not 1")

### test of SysDescription class ###############################################

class testSysDescription:
    def setup(self):
        self.sys110 = stodynprog.SysDescription((1,1,0), stationnary=True, name='sys110')
        self.sys111 = stodynprog.SysDescription((1,1,1), stationnary=True, name='sys111')
        
    def test_attributes(self):
        'SysDescription object attributes'
        assert_true(self.sys111.stationnary)
        
        assert_true(self.sys111.stochastic)
        assert_true(not self.sys110.stochastic)
        
        assert_equal(self.sys111.name, 'sys111')
    
    def test_dyn_function(self):
        # Appropriate dynamical function:
        def dyn3(my_state, my_control, my_perturb):
            pass
        # Assign dynamical function:
        self.sys111.dyn = dyn3
        # Check variable names which are read from the dynamical function:
        assert_equal(self.sys111.state, ['my_state'])
        assert_equal(self.sys111.control, ['my_control'])
        assert_equal(self.sys111.perturb, ['my_perturb'])
        
        # Unappropriate dynamical functions:
        def dyn2(x,u):
            pass
        def dyn4(x,y,u,w):
            pass
        with assert_raises(ValueError):
            self.sys111.dyn = dyn2
        with assert_raises(ValueError):
            self.sys111.dyn = dyn4
    
    def test_print_summary(self):
        # nothing to test really, expect that it runs without error:
        self.sys111.print_summary()
        assert_true(True)
