#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Test the dolointerpolation module for multilinear interpolation

Pierre Haessig — June 2013
"""

from __future__ import division, print_function, unicode_literals
from nose.tools import assert_true
import numpy as np

from stodynprog.dolointerpolation import MultilinearInterpolator
from stodynprog.dolointerpolation.multilinear_cython import multilinear_interpolation

class test_multilinear_interpolation:
    'test the low-level `multilinear_interpolation` routine'
    def test_1D(self):
        '''test a simple 1D linear interpolation'''
        smin = np.array([0.])
        smax =  np.array([2.])
        orders =  np.array([3]) # ie. grid = [0,1,2]
        grid = np.linspace(smin[0], smax[0], orders[0])
        def f(x):
            'function to interpolate'
            return x**2
        values = f(grid)
        values = np.ascontiguousarray(np.atleast_2d(values))
        
        eval_pts = np.linspace(smin[0], smax[0], 5)
        eval_pts =  np.ascontiguousarray(np.atleast_2d(eval_pts))
        print(eval_pts)
        
        interp_values = multilinear_interpolation(smin, smax, orders,
                                                  values, eval_pts)
        print('interpolated values:')
        print(interp_values)
        interp_expected = np.array([0,0.5,1,2.5,4]) # linear interp. "by hand"
        print('expected interpolated values:')
        print(interp_expected)
        assert_true(np.all(np.abs(interp_values-interp_expected) < 1e-10))


class testMultilinearInterpolator:
    'test the high-level `MultilinearInterpolator` class'
    def test_R2R2(self):
        '''test interpolation with a R² to R² function'''
        smin = [1,1]
        smax = [2,2]
        orders = [5,5]
        def f(x):
            '''test function to interpolate
            f: from R² to R²
               x, y -> √(x² + y²), ∛(x³ + y³)
            '''
            return np.row_stack([
                                 np.sqrt( x[0,:]**2 + x[1,:]**2 ),
                                 np.power( x[0,:]**3 + x[1,:]**3, 1.0/3.0 )
                                ])

        # Just call the function
        x = np.array([[0,0]]).T
        f(x)

        print('Building the interpolator')
        interp = MultilinearInterpolator(smin,smax,orders)
        interp.set_values( f(interp.grid) )

        ### Evaluate the interpolation
        # 1) Evaluation points on the grid:
        grid_points = np.array([[1,1],
                                [1,2],
                                [2,1],
                                [2,2]]).T
        # 2) N random points in [1,2]x[1,2]
        N = 6
        random_points = np.random.random( (2, N) )+1

        pts_list = [grid_points, random_points]
        tolerance_list = [1e-9, 0.01]
        for pts, tol in zip(pts_list, tolerance_list):
            interpolated_values = interp(pts)
            exact_values = f(pts)

            print('Comparison interpol vs. exact')
            print(
            np.hstack((interpolated_values.T,
                       exact_values.T
                     )) )

            print('Differences')
            differences = interpolated_values-exact_values
            print(differences.T)
            assert_true(np.all(np.abs(differences) < tol))


#class MlinInterpolator:
#    '''Multilinear interpolation class
#    wrapping Pablo Winant's Cython interpolation routine
#    
#    Note : API of this class is different from Pablo Winant's MultilinInterpolator
#    '''
#    def __init__(self, *x_grid):
#        self.ndim = len(x_grid)
#        self._xmin = np.array([x[0]  for x in x_grid])
#        self._xmax = np.array([x[-1] for x in x_grid])
#        self._xshape = np.array([len(x) for x in x_grid], dtype=np.int)
#        
#        self.values = None
#        
#    def set_values(self,values):
#        assert values.ndim == self.ndim
#        assert values.shape == tuple(self._xshape)
#        self.values = np.ascontiguousarray(np.atleast_2d(values.ravel()))
#    
#    
#    def __call__(self, *x_interp):
#        '''evaluate the interpolated function at coordinates `x_interp`
#        
#        output shape is the shape of broadcasted coordinate inputs.
#        '''
#        assert len(x_interp) == self.ndim
#        # Prepare the interpolated coordinates array
#        x_mesh = np.broadcast_arrays(*x_interp)
#        shape = x_mesh[0].shape
#        x_stack = np.row_stack([x.astype(float).ravel() for x in x_mesh])
#        #
#        a = multilinear_interpolation(self._xmin, self._xmax, self._xshape,
#                                      self.values, x_stack)
#        a = a.reshape(shape)
#        return a

#N = 10
#xg = np.linspace(0,1,N)
#yg = np.linspace(-1,1,N+1)
#zg = np.linspace(2,3,N+2)
#val = np.zeros((N, N+1, N+2))

#minterp = MlinInterpolator(xg, yg, zg)
#minterp.set_values(val)

#minterp(0,0,0)
#minterp([0,1],np.array([[0,1]]).T,0)
