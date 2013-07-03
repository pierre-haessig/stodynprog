#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Example of an Inventory Control problem for the documentation

This code closely matches the code chunks of example_inventory.rst

Plots are in seperate file (for embedded plot generation)
 * example_inventory_plot_policy.py
 * example_inventory_plot_simulation.py

Pierre Haessig — July 2013
"""

from __future__ import division, print_function, unicode_literals
import numpy as np
import scipy.stats as stats
import matplotlib.pylab as plt

import sys
try:
    import stodynprog
except ImportError:
    sys.path.append('..')
    import stodynprog

#reload(stodynprog)

from stodynprog import SysDescription
invsys = SysDescription((1,1,1), name='Shop Inventory')

def dyn_inv(x, u, w):
    'dynamical equation of the inventory stock `x`. Returns x(k+1).'
    return (x + u - w,)
# Attach the dynamical equation to the system description:
invsys.dyn = dyn_inv

demand_values = [0,   1,   2,   3]
demand_proba  = [0.2, 0.4, 0.3, 0.1]
demand_law = stats.rv_discrete(values=(demand_values, demand_proba))
demand_law = demand_law.freeze()

demand_law.pmf([0, 3]) # Probality Mass Function
demand_law.rvs(10) # Random Variables generation




invsys.perturb_laws = [demand_law] # a list, to support several perturbations

def admissible_orders(x):
       'interval of allowed orders U(x_k)'
       U1 = (0, 10)
       return (U1, ) # tuple, to support several controls
# Attach it to the system description.
invsys.control_box = admissible_orders


### Cost description g = r(x) + c.u
(h,p,c) = 0.5, 3, 1
def op_cost(x,u,w):
   'operational cost of the shop'
   holding = x*h
   shortage = -x*p
   order = u*c
   return np.where(x>0, holding, shortage) + order

# Test of the cost function
op_cost(1,1,0)
# Vectorized cost computation capability (required):
op_cost(np.array([-2,-1,0,1,2]),1,0)

invsys.cost = op_cost

#invsys.print_summary()

print('Invertory Control with (h={:.1f}, p={:.1f}, c={:.1f})'.format(h,p,c))

### DP Solver ##################################################################
from stodynprog import DPSolver
dpsolv = DPSolver(invsys)

# discretize the state space
xmin, xmax = (-3,6)
N_x = xmax-xmin+1 # number of states
dpsolv.discretize_state(xmin, xmax, N_x)

# discretize the perturbation
N_w = len(demand_values)
dpsolv.discretize_perturb(demand_values[0], demand_values[-1], N_w)
# control discretization step:
dpsolv.control_steps=(1,) #

#dpsolv.print_summary()

### Value iteration
J_0 = np.zeros(N_x)
# first iteration
J,u = dpsolv.value_iteration(J_0)
print(u[...,0])
# A few more iterations
J,u = dpsolv.value_iteration(J)
print(u[...,0])
J,u = dpsolv.value_iteration(J)
print(u[...,0])
J,u = dpsolv.value_iteration(J)
print(u[...,0])
J,u = dpsolv.value_iteration(J)
print(u[...,0])
J,u = dpsolv.value_iteration(J)
print(u[...,0])

print('stock + order (x+u):')
x_grid = dpsolv.state_grid[0]
print(x_grid + u[...,0])

