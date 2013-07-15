#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Asssess the performance of the storage management for SEAREV smoothing
based on P_grid statistics

Pierre Haessig — July 2013
"""

from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt

from searev_data import load, searev_power, dt, power_max
try:
    import stodynprog
except ImportError:
    import sys
    sys.path.append('..')
    import stodynprog

from storage_simulation import storage_sim, P_sto_law_lin

def cost(P_grid):
    '''penalty on the power injected to the grid
    penal = (P_grid/power_max)**2
    '''
    penal = (P_grid/power_max)**2
    return penal


### Compare with optimized policy:
# Load optimized trajectory:
import pickle
with open('P_sto_law.dat') as f:
    P_sto_law_opt = pickle.load(f)

P_sto_law = P_sto_law_opt
## Enable saturation:
#sat = lambda A,l : (A if A>-l else -l) if A<l else l
#a_sat = 0.5
#print('accel saturation at {:.3f}'.format(a_sat))
#P_sto_law = lambda E,S,A : P_sto_law_opt(E,S,sat(A, a_sat))


datafiles = ['Em_1.txt', 'Em_2.txt', 'Em_3.txt']
std_list = []
mse_list = []


for fname in datafiles:
    t, elev, angle, speed, torque, accel = load(fname)
    P_prod = speed*torque/1e6 # [MW]

    # Run two simulations:
    _, P_grid_lin, _ = storage_sim(speed, accel, P_prod, P_sto_law_lin)
    _, P_grid_opt, _ = storage_sim(speed, accel, P_prod, P_sto_law, check_bounds=False)
    
    
    std_nosto = P_prod.std()
    mse_nosto = cost(P_prod).mean()
    
    std_lin = P_grid_lin.std()
    mse_lin = cost(P_grid_lin).mean()
    print('linear control:    {:.3f} std, {:.4f} mse'.format(std_lin, mse_lin))
    std_opt = P_grid_opt.std()
    mse_opt = cost(P_grid_opt).mean()
    print('optimized control: {:.3f} std, {:.4f} mse'.format(std_opt, mse_opt))


    # Improvement:
    std_change = (std_opt - std_lin)/std_lin
    mse_change = (mse_opt - mse_lin)/mse_lin
    print('criterion reduction: {:.0%} std, {:.0%} mse'.format(std_change, mse_change))
    
    std_list.append((std_nosto, std_lin, std_opt))
    mse_list.append((mse_nosto, mse_lin, mse_opt))
    
    print()

# Convert to arrays:
s = np.array(std_list)
m = np.array(mse_list)

### Plot
plt.figure(figsize=(4.5,3.5))

plt.plot(s.T, 'bo-', lw=0.2)
#plot(m.T, 'gD--')

plt.title('Benefits of storage control optimization')
plt.xticks([0,1,2], ['no storage',
                     'linear\ncontrol',
                     'optimized\ncontrol'])
plt.xlim(-0.5, 2.5)
plt.ylim(ymin=0)
plt.ylabel('$P_{grid}$ standard deviation (MW)')
plt.grid(False, axis='x')

# Annotate:
for i,fname in enumerate(datafiles):
    plt.text(1+0.1, s[i][1], fname,
             verticalalignment='center')

plt.tight_layout()
plt.show()
