#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Plot the optimal ordering policy of the Inventory Control problem
"""

from example_inventory import dpsolv, u
from pylab import *

# grid of the state space:
xr = dpsolv.state_grid[0]

plot(xr, u, '-x')
title('Optimal ordering policy')
xlabel('Stock $x_k$')
ylabel('Number of items to order $u_k$')
ylim(-0.5, u.max()+0.5)
show()
