#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Simulate & Plot a trajectory of the Inventory Control Problem
(see example_inventory.py)
"""

from pylab import *

import example_inventory
reload(example_inventory)
from example_inventory import invsys, dpsolv, u, h,p,c

# Number of instants to simulate
N = 20

demand_law = invsys.perturb_laws[0]

stock = np.zeros(N+1)
order = np.zeros(N)
np.random.seed(0)
demand = demand_law.rvs(N)

x_grid = dpsolv.state_grid[0]
def order_pol(x):
    'ordering policy, for a given stock level `x`'
    x_ind = np.where(x_grid==x)[0]
    return u[x_ind, 0]

# Simpler alternative:
order_pol = dpsolv.interp_on_state(u[...,0])

x0 = 0 # Initial stock
for k in range(N):
    order[k] = order_pol(stock[k])
    stock[k+1] = invsys.dyn(stock[k], order[k], demand[k])[0]

### Plot
t = arange(N)
t1 = arange(N+1)

figure(0, figsize=(8,3))

plot(t1, stock, '-d', label='stock $x_k$')
plot(t, order, '-x', label='order $u_k$')
plot(t, demand, '-+', label='demand $w_k$')
hlines(0, 0, N, colors='gray')

title('Simulated trajectory of the Invertory Control (h={:.1f}, p={:.1f}, c={:.1f})'.format(h,p,c))
xlabel('time $k$')
ylabel('Number of items')
xlim(-0.5, N+5.5)
ylim(-1.2, 3.2)
legend(loc='upper right')

tight_layout()
show()
