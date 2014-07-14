#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Compute an optimal storage control policy
supposing a perfect knowledge of the future inputs.

If run as a script, simulate and plot the optimal trajectory.

Pierre Haessig — November 2013
"""

from __future__ import division, print_function, unicode_literals
import numpy as np

from stodynprog import SysDescription, DPSolver
from storagesim import AR1
import itertools


### Parameters of the problems:

dt = 1 # [h]

# AR(1) process to generate input
P_req_ar1 = AR1(0, scale = 1., # [MW]
                corr = 0.8, # AR(1) correlation coeff.
                n_sim=1)

p = {
'E_rated': 20., # [MWh]
'P_rated': 4. * P_req_ar1.scale, # [MW]
'P_req_data': []
}

print('Storage ratings: {:.2f} MWh'.format(p['E_rated']))

# Length of the optimization horizon:
T_horiz = 14*24
#T_horiz = 1464

## Generate the input:
print('generate P_req data:', end=' ')
print('gaussian AR(1) noise with std={:.2f} and corr={:.2f}'.format(
       P_req_ar1.scale, P_req_ar1.corr))


def generate_input(seed):
    '''fill p['P_req_data'] with an AR(1) sample'''
    P_req_ar1.seed = seed
    P_req_ar1.restart()
    # generate samples:
    P_req = list(itertools.islice(P_req_ar1, T_horiz))
    # reshape to a flat 1D array:
    P_req = np.array(P_req).reshape(-1)
    p['P_req_data'] = P_req

generate_input(seed=2)

def dyn_sto(k, E_sto, P_sto):
    '''state transition of the "deterministic storage" system

    State variables:
    * E_sto

    Control:
    * P_sto
    '''
    # Stored energy:
    E_sto_n = E_sto + P_sto*dt
    return (E_sto_n, )

def admissible_controls(k, E_sto):
    '''set of admissible control U(x_k) of an Energy storage
    Controls is the stored power P_sto
    
    Contrainsts of the Energy Storage are:
    1) Energy stock boundaries : 0 ≤ E(k + 1) ≤ E_rated
    2) Power limitation : -P_rated ≤ P_sto ≤ P_rated
    '''
    # 1) Constraints on P_sto:
    P_neg = np.max(( -E_sto/dt, - p['P_rated']))
    P_pos = np.min(( (p['E_rated']- E_sto)/dt, p['P_rated']))
    U1 = (P_neg, P_pos)
    return (U1, )

def cost_model(k, E_sto, P_sto):
    '''penalty on the power that is not absorbed
    P_dev = P_req - P_sto
    penal = P_dev**2
    '''
    P_req = p['P_req_data'][k]
    P_dev = P_req - P_sto
    penal = P_dev**2
    return penal

### Create the system description:
sto_sys = SysDescription((1,1,0), name='Deterministic Storage', stationnary=False)
sto_sys.dyn = dyn_sto
sto_sys.control_box = admissible_controls
sto_sys.cost = cost_model

sto_sys.print_summary()

### Create the DP solver:
dpsolv = DPSolver(sto_sys)
# discretize the state space
N_E = 100

E_grid = dpsolv.discretize_state(0, p['E_rated'], N_E)[0]
dpsolv.control_steps=(.001,)

dpsolv.print_summary()

if __name__ == '__main__':
    ### Solve the problem:
    # Zero final cost
    J_fin = np.zeros(N_E)
    # Solve the problem:
    J, pol = dpsolv.bellman_recursion(T_horiz, J_fin)

    # RMS cost from the cost function:
    J_l2_empty = np.sqrt(J[0,0]/T_horiz)
    J_l2_mid = np.sqrt(J[0,N_E//2]/T_horiz)
    J_l2_full = np.sqrt(J[0,-1]/T_horiz)
    print('output RMS deviation from cost function')
    print('  if SoE(0)=0  : {:.4f}'.format(J_l2_empty))
    print('  if SoE(0)=0.5: {:.4f}'.format(J_l2_mid))
    print('  if SoE(0)=1  : {:.4f}'.format(J_l2_full))
    print('  effect of SoE(0): {:.2%}'.format((J_l2_full- J_l2_empty)/J_l2_empty
    ))

    pol_sto = pol[..., 0]


    ### Simulation: ###
    N_sim = T_horiz
     # Time vector
    k_range  = np.arange(N_sim)
    k_range_x= np.arange(N_sim+1)
    # State variables
    E = np.zeros(N_sim+1)
    E[0] = p['E_rated']/2

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
    E_full = np.ma.array(E, mask =  (E<p['E_rated']*0.9999))
    E_empty = np.ma.array(E, mask = (E>p['E_rated']*0.0001))
    # Deviation from commitment:
    P_dev = p['P_req_data'] - P_sto

    P_req_l2 = np.sqrt(np.mean(p['P_req_data']**2))
    P_dev_l2 = np.sqrt(np.mean(P_dev**2))
    print('input  RMS deviation: {:.4f}'.format(P_req_l2))
    print('output RMS deviation: {:.4f}'.format(P_dev_l2))

    ### Plot:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2,1, sharex=True)

    ax[0].set_title('Power flows')
    ax[0].plot(k_range, p['P_req_data'], 'b-',  label='$P_{req}$')
    ax[0].plot(k_range, P_sto, 'c-',  label='$P_{sto}$')
    ax[0].plot(k_range, P_dev, 'r-',  label='$P_{dev}$')
    ax[0].legend()

    ax[1].set_title('Stored energy')
    ax[1].plot(k_range_x, E, 'b-',  label='$E_{sto}$')
    ax[1].plot(k_range_x, E_full, 'D-', color='red', label='full')
    ax[1].plot(k_range_x, E_empty, 'D-', color='orange', label='empty')

    plt.show()
    
    ## Enhanced plot: (using storagesim.plotting)
    from storagesim.plotting import plot_trajectory, trajectory_params
    
    SoE = E/p['E_rated']
    t = k_range/24.
    trajectory_params['show_dev_stat'] = 'L2'
    
    fig = plot_trajectory(t, SoE[1:], p['P_req_data'], P_sto,
                          pow_max = 3.5*P_req_ar1.scale,
                          params = trajectory_params)
    fig_fname = 'traj_det_E{:.1f}_corr{:.1f}'.format(p['E_rated'], P_req_ar1.corr)
    fig.savefig(fig_fname + '.pdf')
    fig.savefig(fig_fname + '.png', dpi=150)
