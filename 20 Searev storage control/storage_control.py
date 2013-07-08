#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Compute an optimal storage control policy
to smooth out the SEAREV power production fluctuations

Pierre Haessig — June 2013
"""

from __future__ import division, print_function, unicode_literals
import sys
from datetime import datetime
import numpy as np
import scipy.stats as stats
import matplotlib as mpl
import matplotlib.pyplot as plt

# Load Searev model data:
from searev_data import searev_power, power_max, dt

# Tweak how images are plotted with imshow
mpl.rcParams['image.interpolation'] = 'none' # no interpolation
mpl.rcParams['image.origin'] = 'lower' # origin at lower left corner
mpl.rcParams['image.aspect'] = 'auto'


try:
    from stodynprog import SysDescription, DPSolver
except ImportError:    
    sys.path.append('..')
    from stodynprog import SysDescription, DPSolver


### SEAREV+storage dynamics description

# Searev AR(2) model at 0.1 s :
c1 = 1.9799
c2 = -0.9879
innov_std = 0.00347
innov_law = stats.norm(loc=0, scale=innov_std)

# Storage rated energy and power:
E_rated = 10 # [MJ]
P_rated = 1.1 # [MW]
a = 0.00 # loss factor

print('Storage ratings: {:.2f} MW / {:.2f} MJ ({:.2f} kWh)'.format(P_rated,
 E_rated, E_rated/3.6))

def dyn_searev_sto(E_sto, Speed, Accel, P_sto, innov):
    '''state transition of the "SEAREV + storage" system
    
    State variables :
    * E_sto
    * Speed
    * Accel
    
    Control:
    * P_sto
    '''
    # Stored energy:
    E_sto_n = E_sto + (P_sto - a*abs(P_sto))*dt
    # AR(2) model of the SEAREV:
    Speed_n = (c1+c2)*Speed   - dt*c2*Accel + innov
    Accel_n = (c1+c2-1)/dt*Speed - c2*Accel + innov/dt
    return (E_sto_n, Speed_n, Accel_n)

def admissible_controls(E_sto, Speed, Accel):
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

def cost_model(E_sto, Speed, Accel, P_sto, innov):
    '''penalty on the power injected to the grid
    P_grid = P_prod - P_sto
    penal = (P_grid/power_max)**2
    '''
    P_prod = searev_power(Speed)
    P_grid = P_prod - P_sto
    penal = (P_grid/power_max)**2
    return penal
cost_label = 'quadratic cost'

### Create the system description:
searev_sys = SysDescription((3,1,1), name='Searev + Storage')
searev_sys.dyn = dyn_searev_sto
searev_sys.control_box = admissible_controls
searev_sys.cost = cost_model
searev_sys.perturb_laws = [innov_law]

#searev_sys.print_summary()

### Create the DP solver:
dpsolv = DPSolver(searev_sys)
# discretize the state space
N_E = 31
N_S = 31
N_A = 31
S_min, S_max = -4*.254, 4*0.254
A_min, A_max = -4*.227, 4*.227
x_grid = dpsolv.discretize_state(0, E_rated, N_E,
                                 S_min, S_max, N_S,
                                 A_min, A_max, N_A)
E_grid, S_grid, A_grid = x_grid
# discretize the perturbation
N_w = 9
dpsolv.discretize_perturb(-3*innov_std, 3*innov_std, N_w)
# control discretization step:
dpsolv.control_steps=(.01,)

dpsolv.print_summary()


# An heuristic control law:
def P_sto_law_lin(E_sto, Speed, Accel):
    '''linear storage control law'''
    P_prod = searev_power(Speed)
    P_grid = P_rated*E_sto/E_rated
    return P_prod - P_grid

###############################################################################
### Optimization of the storage control law with policy iteration

# A policy to start with:
pol_lin = P_sto_law_lin(*dpsolv.state_grid_full)
pol_lin = pol_lin[..., np.newaxis]


### Look at the convergence of policy evaluation
n_val = 1000
J,J_ref = dpsolv.eval_policy(pol_lin, n_val, True, J_ref_full=True)

plt.figure('policy evaluation convergence')
plt.plot(J_ref)
ref_lim = J_ref[-1]
plt.hlines((ref_lim*.99, ref_lim*1.01),  0, n_val-1 , label='limit +/- 1 %',
           linestyles='dashed', alpha=0.5)
plt.title('Convergence of policy evaluation (grid {:d},{:d},{:d})'.format(N_E, N_S, N_A))
plt.xlabel('Iterations of policy evaluation')
plt.ylabel('Reference cost of linear policy')
plt.show()

#print('reference cost after {:d} iterations of policy evaluation: {:3f}'.format(n_val, ref_lim))

### Policy iteration:
r = 0.
n_pol = 5
(J, r), pol = dpsolv.policy_iteration(pol_lin, n_val, n_pol, rel_dp=True)
pol_fname = 'pol_E{:d}_grid3131_iter{:d}.npy'.format(E_rated, n_pol)
np.save(pol_fname, pol); print('saving {:s}.npy'.format(pol_fname))
pol = np.load(pol_fname)


print('reference cost after {:d} policy improvements: {:3f}'.format(n_val, r))





# Extract the P_sto law:
pol_sto = pol[..., 0]


#### Effect of the state discretization ########################################
#N_grid_list = [9,10,11, 19,20,21, 30,31, 51, 61, 71]

#J_ref_list = []
#for N_grid in N_grid_list:
#    print('discretization pts: {:d}^3'.format(N_grid))
#    dpsolv.discretize_state(0, E_rated, N_grid,
#                            S_min, S_max, N_grid,
#                            A_min, A_max, N_grid)
#    pol_lin = P_sto_law_lin(*dpsolv.state_grid_full)
#    pol_lin = pol_lin[..., np.newaxis]
#    J,J_ref = dpsolv.eval_policy(pol_lin, n_val, True)
#    J_ref_list.append(J_ref)


#plt.figure()
#plt.plot(N_grid_list, J_ref_list, '-x')
#plt.hlines(.061, 0, N_grid_list[-1], label='true cost ?')
#plt.title('Effect of the state discretization')
#plt.xlabel('size of the discretized grid (same in all three dimensions)')
#plt.ylabel('Reference cost of linear policy')



### Trajectory simulation ######################################################
N_sim = 10**4
seed = 0


print('simulating a trajectory along {:d} instants ({:.2f} s)...'.format(N_sim, N_sim*dt))
# Initial state conditions x(0) :
E_sto_0, Speed_0, Accel_0 = (E_rated/3, 0, 0)


# State variables
E_sto = np.zeros(N_sim+1)
Speed = np.zeros(N_sim+1)
Accel = np.zeros(N_sim+1)
E_sto[0] = E_sto_0
Speed[0] = Speed_0
Accel[0] = Accel_0


#P_sto_law = P_sto_law_lin
# use optimal control law :
P_sto_law = dpsolv.interp_on_state(pol_sto)

# Load another policy:
#P_sto_law = dpsolv.interp_on_state(np.load('storage control/u.npy')[...,0])


P_sto = np.zeros(N_sim)


# Output variable:
P_prod = np.zeros(N_sim)
P_grid = np.zeros(N_sim)

# draw a disturbance sequence:
np.random.seed(seed)
w = innov_law.rvs(N_sim)

# Time vector
k_range  = np.arange(N_sim)
t = k_range*dt
t_x = np.arange(N_sim+1)*dt

# Simulation loop:
for k in k_range:
    # Searev Power :
    P_prod[k] = searev_power(Speed[k])
    # Control computation:
    P_sto[k] = P_sto_law(E_sto[k], Speed[k], Accel[k])
    
    P_grid[k] = P_prod[k] - P_sto[k]
    # State evolution:
    E_sto[k+1], Speed[k+1], Accel[k+1] = searev_sys.dyn(E_sto[k], Speed[k], Accel[k],
                                                 P_sto[k], w[k])

# Compute state variables derivatives:
E_full = np.ma.array(E_sto, mask =  (E_sto<E_rated*0.9999))
E_empty = np.ma.array(E_sto, mask = (E_sto>E_rated*0.0001))

# Power delivered to the grid
P_grid = P_prod - P_sto

cost = cost_model(E_sto[:-1], Speed[:-1], Accel[:-1],
                  P_sto, w)


print('average cost of the trajectory : {:f}'.format(cost.mean()))
print('P_grid mean : {:.4f} MW'.format(P_grid.mean()) )
print('P_grid std  : {:.4f} MW'.format(P_grid.std()) )
print('P_prod mean : {:.4f} MW'.format(P_prod.mean()) )
print('P_prod std  : {:.4f} MW'.format(P_prod.std()) )

fig = plt.figure('trajectories')
ax = fig.add_subplot(311, title='Searev with storage', ylabel='Power (MW)')

# 1) P_prod, P_grid
plt.plot(t, P_prod, color='gray')
plt.plot(t, P_grid)
ax.hlines(P_prod.mean(), t[0], t[-1], color='white', zorder=3, alpha=0.5)

ax = fig.add_subplot(312, sharex=ax, ylabel='$P_{sto}$ (MW)')
plt.plot(t, P_sto)


ax = fig.add_subplot(313, sharex=ax, xlabel='time (s)', ylabel='$E_{sto}$ (MJ)')
plt.plot(t_x, E_sto)


##### Plot the policy

## P_prog range :
#P_p = searev_power(S_grid).reshape(1,-1)
#P_s = ctrl_sto[0,:,:,N_A//2]
#P_g = P_p - P_s


#plt.figure('control law', figsize=(12,6))
#plt.subplot(121)
#plt.imshow(P_s, extent=(S_min, S_max, 0, E_rated), vmin=-P_rated, vmax = P_rated)
#plt.title('P_sto')
#plt.colorbar()

#plt.subplot(122)
#plt.imshow(P_g, extent=(S_min, S_max, 0, E_rated), vmin=-P_rated, vmax = P_rated)
#plt.title('P_grid = P_prod-P_sto')
#plt.colorbar()


## Mayavi:
#from volume_slicer import VolumeSlicer
#P_p3 = searev_power(S_grid).reshape(1,-1,1)
#m = VolumeSlicer(data=J_opt[0, :,:,:])
#m = VolumeSlicer(data=P_p3 - ctrl_sto[0, :,:,:])
#m.configure_traits()

## Contour 3D:
#from mayavi import mlab
#x,y,z = np.broadcast_arrays(E_grid.reshape(-1,1,1)/E_rated*2,
#                            S_grid.reshape(1,-1,1), A_grid)
#c= mlab.contour3d(x,y,z,P_p3 - ctrl_sto[0, :,:,:], contours=10)
## Animation:
#for i in range(N)[::-1]:
#    c.mlab_source.scalars = P_p3 - ctrl_sto[i, :,:,:]


plt.show()
