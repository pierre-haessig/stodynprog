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

# time step:
dt = 0.1

# Searev power-take-off (PTO) parameters
damp = 4.e6 # N/(rad/s)
torque_max = 2e6 # N.m
power_max = 1.1 # MW

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

# Torque command law ("PTO strategy")
def searev_power(speed):
    '''Searev power-take-off as function of speed (rad/s)
    returns P_prod (MW)
    '''
    tor = speed * damp
    # 1) Max torque limitation:
    tor = np.where(tor >  torque_max,  torque_max, tor)
    tor = np.where(tor < -torque_max, -torque_max, tor)
    # 2) Max power limitation:
    P_prod = tor*speed/1e6 # W -> MW
    P_prod = np.where(P_prod > power_max, power_max, P_prod)
    return P_prod

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
dpsolv.control_steps=(.01,) # 0.02 MW step

dpsolv.print_summary()


###############################################################################
### Computation

# Number of DP iterations:
N = 600
# range of instants to solve:
k_range = np.arange(N-1, -1, -1)
print('DP algorithm with {:d} iterations...'.format(N))

### Initialize memory variables:
# a) cost
J_opt = np.zeros((N+1, N_E, N_S, N_A))
# Start with a zero terminal cost:
J_zero = np.zeros((N_E, N_S, N_A))
J_opt[N] = J_zero
# b) control
ctrl_opt = np.zeros((N, N_E, N_S, N_A, 1))

# Reference cost (for *relative* value iteration)
J_ref_opt = np.zeros(N+1)
ref_ind = (N_E//2, N_S//2, N_A//2)
#ref_ind = (0,0)
print('reference state: E_sto = {:.2f} MJ, S = {:.2f} rad/s, A = {:.2f} rad/s^2'.format(
      E_grid[ref_ind[0]], S_grid[ref_ind[0]], A_grid[ref_ind[0]]))


### Computation
t_start = datetime.now()

for k in k_range:
    print('\r  solving instant {:d} (N-{:d})  '.format(k,N-k), end='')
    sys.stdout.flush()
    continue ### Skip the computation ###
    # one step Value iteration algo:
    J_opt[k], ctrl_opt[k] = dpsolv.solve_step(J_opt[k+1], report_time=False)
    # Take a reference state to get the reference cost
    J_ref_opt[k] = J_opt[k][ref_ind]
    # ... and substract this reference cost to keep only the differential cost:
    J_opt[k] -= J_ref_opt[k]
print('')

## Save and load computation results :
#np.savez('dp_arrays.npz', J_opt=J_opt, J_ref_opt=J_ref_opt, ctrl_opt=ctrl_opt)
# Load:
dp_arrays = np.load('dp_arrays_600_31.npz')
J_opt = dp_arrays['J_opt']
J_ref_opt = dp_arrays['J_ref_opt']
ctrl_opt = dp_arrays['ctrl_opt']


## Print some statitics:
exec_time = (datetime.now() - t_start).total_seconds()
print('DP iterations run in {:.1f} s ({:.2f} s/iter)'.format(exec_time, exec_time/N))
print('')

print('Reference cost after {:d} iterations:'.format(N))
print('{:g} {:s}'.format(J_ref_opt[0], cost_label))
if N>=2:
    rel_var = (J_ref_opt[0] - J_ref_opt[1])/J_ref_opt[0]
    print('  relative variation from iteration N-1: {:.2%}'.format(rel_var))

# Compute the range of the differential cost:
cost_range = (J_opt[0].min(), J_opt[0].max(), J_opt[0].max() - J_opt[0].min())
print('Range of the relative cost: {:.4g} to {:.4g}'.format(*cost_range))

print('')

# Extract the P_sto control law:
ctrl_sto = ctrl_opt[...,0]



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

# Control variables
def P_sto_law_lin(E_sto, Speed, Accel):
    '''linear storage control law'''
    P_prod = searev_power(Speed)
    P_grid = P_rated*E_sto/E_rated
    return P_prod - P_grid

P_sto_law = P_sto_law_lin
# use optimal control law :
#P_sto_law = dpsolv.interp_on_state(ctrl_sto[0,:,:,:])

# Load another policy:
#P_sto_law = dpsolv.interp_on_state(np.load('u.npy')[...,0])


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
