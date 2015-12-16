#!/usr/bin/python
# -*- coding: utf-8 -*-

import test_dolointerp, test_stodynprog

def run(verbose=False):
    '''run the unit tests of stodynprog (using `nose` module)
    
    return True if test run succeeded, False otherwise
    '''
    import nose
    import sys
    print('Stodynprog unit tests:')
    argv = [sys.argv[0]]
    if verbose:
        argv.append('--verbose')
    argv.append(__path__[0])
    success = nose.run(argv=argv)
    return success