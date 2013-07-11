#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Simulate storage policies onto SEAREV time series
(and plot)

* with linear policy
*

Pierre Haessig — July 2013
"""

from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt

from searev_data import load, searev_power, dt
try:
    import stodynprog
except ImportError:    
    sys.path.append('..')
    import stodynprog

# Storage rated energy and power:
E_rated = 10 # [MJ]
P_rated = 1.1 # [MW]
a = 0.00 # loss factor

E_sto_0 = E_rated/3

# Storage Control law:
def P_sto_law_lin(E_sto, Speed, Accel):
    '''linear storage control law'''
    P_prod = searev_power(Speed)
    P_grid = P_rated*E_sto/E_rated
    return P_prod - P_grid

#P_sto_law = dpsolv.interp_on_state(ctrl_sto[0,:,:,:])

#TODO: add the ability to load optimized control laws

# colors
c = {'red':'#df025f',
     'light red':'#ffbfda',
     'purple':'#68006C',
     'steel blue':'#469bd0'
     }


def storage_sim(speed, accel, P_prod, P_sto_law=P_sto_law_lin, E_sto_0 = E_rated/3):
    '''Simple SEAREV storage simulation.
    (without saturation support -> raises an error if storage runs out of bounds)
    
    Returns:
    P_prod, P_sto, P_grid, E_sto
    '''
    N_sim = len(P_prod)
    
    P_sto = np.zeros(N_sim)
    P_grid = np.zeros(N_sim)
    
    # State variables:
    E_sto = np.zeros(N_sim+1)
    E_sto[0] = E_sto_0
    
    k_range  = np.arange(N_sim)
    
    # Simulation loop:
    for k in k_range:
        # Storage Control computation:
        P_sto[k] = P_sto_law(E_sto[k], speed[k], accel[k])
        assert -P_rated <= P_sto[k] <= P_rated
        
        # State evolution:
        E_sto_n = E_sto[k] + (P_sto[k] - a*abs(P_sto[k]))*dt
        assert 0 <= E_sto_n <= E_rated
        E_sto[k+1] = E_sto_n
    # end for
    # Power delivered to the grid
    P_grid = P_prod - P_sto
    return P_sto, P_grid, E_sto


def plot_trajectories(t, P_prod, P_sto, P_grid, E_sto):
    '''Plot a trajectory of the SEAREV+storage system
    (3 panels)
    
    Returns: fig
    '''
    t_x = np.arange(len(t)+1)*dt
    
    fig = plt.figure('trajectories')
    # Create the suplots:
    if len(fig.axes) != 3:
        fig .clear()
        ax1 = fig.add_subplot(311, title='Searev with storage "{}"'.format(fname),
                                   ylabel='Power (MW)')
        ax2 = fig.add_subplot(312, sharex=ax1, ylabel='$P_{sto}$ (MW)')
        ax3 = fig.add_subplot(313, sharex=ax1, xlabel='time (s)',
                                   ylabel='$E_{sto}$ (MJ)')
    else:
        # use existing axes
        ax1, ax2, ax3 = fig.axes
    
    # 1) plot P_prod, P_grid
    ax1.plot(t, P_prod, color=c['light red'], label='$P_{prod}$', zorder=2)
    ax1.plot(t, P_grid, color=c['purple'], label='$P_{grid}$', zorder=3)
    ax1.hlines(P_prod.mean(), t[0], t[-1], label='average',
              color=c['red'], zorder=4, alpha=0.5)
    
    if ax1.get_legend() is None:
        ax1.legend(loc='upper right', prop={'size':10}, borderaxespad=0.)

        # fine tune the legend:
        box = ax1.get_legend().get_frame()
        box.set_linewidth(0.5) # thin border
        box.set_facecolor([1]*3) # white
        box.set_alpha(.7)
    
    # 2) plot P_sto
    ax2.plot(t, P_sto, color=c['steel blue'])
    
    # 3) plot E_sto
    ax3.plot(t_x, E_sto, color=c['steel blue'])
    #plt.hlines([0, E_rated], 0, t[-1], colors='gray')
    ax3.set_ylim(0, E_rated)

    fig.tight_layout()
    plt.show()
    return fig

if __name__ == '__main__':
    ### Load time series:
    fname = 'Em_1.txt'
    t, elev, angle, speed, torque, accel = load(fname)
    P_prod = speed*torque/1e6 # [MW]
    
    # Load optimized trajectory:
    import pickle
    with open('P_sto_law.dat') as f:
        P_sto_law_opt = pickle.load(f)
    
    sat = lambda A,l : (A if A>-l else -l) if A<l else l
    a_sat = accel.std()*2.5
    print('accel saturation at {:.3f}'.format(a_sat))
    P_sto_law = lambda E,S,A : P_sto_law_opt(E,S,sat(A, a_sat))
    
    # Run the simulation:
    P_sto, P_grid, E_sto = storage_sim(speed, accel, P_prod, P_sto_law)
    
    #mpl.rcParams['grid.color'] = (0.66,0.66,0.66, 0.4)
    
    ### Plot the trajectories ###
    plot_trajectories(t, P_prod, P_sto, P_grid, E_sto)
