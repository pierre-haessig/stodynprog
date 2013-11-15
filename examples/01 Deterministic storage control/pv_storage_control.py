#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Compute an optimal storage control policy
for a PV plant
supposing a perfect knowledge of the future inputs.

Pierre Haessig — November 2013
"""

from __future__ import division, print_function, unicode_literals
import sys
from datetime import datetime
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# Tweak how images are plotted with imshow
mpl.rcParams['image.interpolation'] = 'none' # no interpolation
mpl.rcParams['image.origin'] = 'lower' # origin at lower left corner
mpl.rcParams['image.aspect'] = 'auto'


try:
    from stodynprog import SysDescription, DPSolver
except ImportError:
    sys.path.append('../..')
    from stodynprog import SysDescription, DPSolver



# Read the PV production data:
t,P_prod_data = np.loadtxt('pv_prod.csv', skiprows=1, delimiter=',').T

## Storage dynamics description
dt = t[1]-t[0]
# Storage rated energy and power:
E_rated = 2 # [MWh]
P_rated = 1 # [MW]
print('Storage ratings: {:.2f} MW / {:.2f} MWh'.format(P_rated, E_rated))

# storage loss factor
a = 0.0
print('  storage loss factor: {:.1%}'.format(a))

T_horiz = len(P_prod_data)

def dyn_sto(k, E_sto, P_sto):
    '''state transition of the "deterministic storage" system

    State variables:
    * E_sto

    Control:
    * P_sto
    '''
    # Stored energy:
    E_sto_n = E_sto + (P_sto - a*abs(P_sto))*dt
    return (E_sto_n, )

def admissible_controls(k, E_sto):
    '''set of admissible control U(x_k) of an Energy storage
    Controls is the stored power P_sto
    
    Contrainsts of the Energy Storage are:
    1) Energy stock boundaries : 0 ≤ E(k + 1) ≤ E_rated
    2) Power limitation : -P_rated ≤ P_sto ≤ P_rated
    '''
    # 1) Constraints on P_sto:
    P_neg = np.max(( -E_sto/(1+a)/dt, -P_rated))
    P_pos = np.min(( (E_rated - E_sto)/(1-a)/dt, P_rated))
    U1 = (P_neg, P_pos)
    return (U1, )

def cost_model(k, E_sto, P_sto):
    '''penalty on the power injected to the grid
    P_grid = P_prod - P_sto
    '''
    P_prod = P_prod_data[k]
    P_grid = P_prod - P_sto
    P_grid_over = np.where(P_grid > 0.4, P_grid-0.4, 0)
    P_grid_neg = np.where(P_grid < 0, P_grid, 0)
    penal = P_grid_over**2 + P_grid_neg**2 + 0*P_sto**2
    return penal

### Create the system description:
sto_sys = SysDescription((1,1,0), name='Deterministic Storage for PV', stationnary=False)
sto_sys.dyn = dyn_sto
sto_sys.control_box = admissible_controls
sto_sys.cost = cost_model

sto_sys.print_summary()

### Create the DP solver:
dpsolv = DPSolver(sto_sys)
# discretize the state space
N_E = 50

E_grid = dpsolv.discretize_state(0, E_rated, N_E)[0]
dpsolv.control_steps=(.001,)

dpsolv.print_summary()

J_fin = np.zeros(N_E)

J, pol = dpsolv.bellman_recursion(T_horiz, J_fin)

pol_sto = pol[..., 0]

### Plot the policy
plt.figure('policy')
plt.subplot(111, title='$P_{grid}(t,E)$ policy', xlabel='time', ylabel='$E_{sto}$')
plt.imshow(P_prod_data - pol_sto.T, extent=(0, (T_horiz-1)/24, 0, 2))

#plt.plot(t/24., E[:-1], 'k-', alpha=0.5, label='$E_{sto}$')
#plt.plot(t/24., P_prod_data+E_rated, label='$P_{prod}$')
plt.colorbar()


#### Simulation: ####
N_sim = T_horiz
 # Time vector
k_range  = np.arange(N_sim)
k_range_x= np.arange(N_sim+1)
# State variables
E = np.zeros(N_sim+1)
E[0] = E_rated/2

# Control variables
P_sto = np.zeros(N_sim)


# Simulation loop:
for k in k_range:
    # Control computation:
    P_sto_law = dpsolv.interp_on_state(pol_sto[k])
    P_sto[k] = P_sto_law(E[k])
    # State evolution:
    E[k+1], = sto_sys.dyn(k, E[k], P_sto[k])

# Compute state variables derivatives:
E_full = np.ma.array(E, mask =  (E<E_rated*0.9999))
E_empty = np.ma.array(E, mask = (E>E_rated*0.0001))
# Deviation from commitment:
P_grid = P_prod_data - P_sto


P_grid_l2 = np.sqrt(np.mean(P_grid**2))
print('RMS deviation: {:.4f}'.format(P_grid_l2))

### Plot:
fig, ax = plt.subplots(2,1, sharex=True)

ax[0].set_title('Power flows of the PV-storage system')
ax[0].plot(k_range, P_prod_data, 'b-',  label='$P_{prod}$')
ax[0].plot(k_range, P_sto, 'c-',  label='$P_{sto}$')
ax[0].plot(k_range, P_grid, 'r-',  label='$P_{grid}$')
ax[0].legend()

ax[1].set_title('Stored energy')
ax[1].plot(k_range_x, E, 'b-',  label='$E_{sto}$')
ax[1].plot(k_range_x, E_full, 'D-', color='red', label='full')
ax[1].plot(k_range_x, E_empty, 'D-', color='orange', label='empty')

plt.show()
