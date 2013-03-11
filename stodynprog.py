#!/usr/bin/python
# -*- coding: UTF-8 -*-
""" Stochastic Dynamic Programming library

Implements naive methods of Dynamic Programming (Value Iteration)
to solve *simple* Optimal Stochastic Control problems

classes : SysDescription, DPSolver
"""

from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
import inspect


def _zero_cost(*x):
    '''zero cost function g(x), used as default terminal cost'''
    return 0.

class SysDescription(object):
    def __init__(self, dims, stationnary=True, name=''):
        '''Description of a Dynamical System in the view of optimal (stochastic)
        control, using Dynamic Programming approach.
        
        Each system basically has
         * a dynamics function x_{k+1} = f_k(x_k, u_k, w_k)
         * an instant cost function g_k(x_k, u_k, w_k)
         
        The sum over instants of g_k is the total cost J which is to be minimized
        by choosing the control policy
        '''
        self.name = name
        self.stationnary = bool(stationnary)
        
        if len(dims) == 3:
            dim_state, dim_control, dim_perturb = dims
        elif len(dims) == 2:
            dim_state, dim_control = dims
            dim_perturb = 0
        else:
            raise ValueError('dims tuple should be of len 2 or 3')
        
        self.state = ['x{:d}'.format(i+1) for i in range(dim_state)]
        self.control = ['u{:d}'.format(i+1) for i in range(dim_control)]
        self.perturb = ['w{:d}'.format(i+1) for i in range(dim_perturb)]
        
        # Expected signature length of dyn and cost functions:
        self._signature_length = dim_state + dim_control + dim_perturb
        if not self.stationnary:
            # for unstationnary systems, instant `k` is provided as 1st argument
            self._signature_length += 1
        
        # Dynamics and Cost functions (to be set separately)
        self._dyn = None
        self._cost = None
        self._terminal_cost = _zero_cost
        self._perturb_laws = None
    
    
    @property
    def stochastic(self):
        '''is the system stochastic or deterministic ?'''
        return len(self.perturb) > 0
    
    @property
    def dyn(self):
        '''dynamics function x_{k+1} = f_k(x_k, u_k, w_k)'''
        return self._dyn
    
    @dyn.setter
    def dyn(self, dyn):
        '''sets the dynamics function'''
        # Check the signature length:
        dyn_args = inspect.getargspec(dyn).args
        if not len(dyn_args) == self._signature_length:
            raise ValueError('dynamics function should accept '
                             '{:d} args instead of {:d}'.format(
                             self._signature_length, len(dyn_args)))
        self._dyn = dyn
        
        # Read the variable names from the signature of `dyn`
        if not self.stationnary:
            # drop the first argument
            dyn_args = dyn_args[1:]
        self.state = dyn_args[0:len(self.state)]
        dyn_args = dyn_args[len(self.state):] # drop state variables
        self.control = dyn_args[0:len(self.control)]
        dyn_args = dyn_args[len(self.control):]  # drop control variables
        self.perturb = dyn_args[0:len(self.perturb)]
        
    @property
    def cost(self):
        '''cost function g_k(x_k, u_k, w_k)'''
        return self._cost
    
    @cost.setter
    def cost(self, cost):
        '''sets the cost function'''
        # Check the signature length:
        cost_args = inspect.getargspec(cost).args
        if not len(cost_args) == self._signature_length:
            raise ValueError('cost function should accept '
                             '{:d} args instead of {:d}'.format(
                             self._signature_length, len(cost_args)))
        self._cost = cost
    
    @property
    def terminal_cost(self):
        '''terminal cost function g(x_K)'''
        return self._terminal_cost
    
    @terminal_cost.setter
    def terminal_cost(self, cost):
        '''sets the terminal cost function'''
        # Check the signature length:
        cost_args = inspect.getargspec(cost).args
        if not len(cost_args) == len(self.state):
            raise ValueError('cost function should accept '
                             '{:d} args instead of {:d}'.format(
                             len(self.state), len(cost_args)))
        self._terminal_cost = cost
    
    @property
    def perturb_laws(self):
        '''distribution laws of perturbations `w_k`'''
        return self._perturb_laws
    
    @perturb_laws.setter
    def perturb_laws(self, laws):
        '''distribution laws of perturbations'''
        # Check the number of laws
        if not len(laws) == len(self.perturb):
            raise ValueError('{:d} perturbation laws should be provided'
                             .format(len(self.perturb)))
        self._perturb_laws = laws
    

    def print_summary(self):
        print('System "{:s}"'.format(self.name))
        print('  state vector:   {:s} (dim {:d})'.format(', '.join(self.state),
                                                     len(self.state)) )
        print('  control vector: {:s} (dim {:d})'.format(', '.join(self.control),
                                                       len(self.control)) )
        if self.stochastic:
            print('  perturbation:   {:s} (dim {:d})'.format(', '.join(self.perturb),
                                                             len(self.perturb)) )
        else:
            print('  no perturbation')
    # end print_summary
    def __repr__(self):
        return '<SysDescription "{:s}" at 0x{:x}>'.format(self.name, id(self))
# end SysDescription

class DPSolver(object):
    def __init__(self, sys):
        '''Dynamic Programming solver based on Value Iteration
        '''
        self.sys = sys
    
    def discretize_state(self, *linspace_args):
        '''create a regular discrete grid for each state variable'''
        assert len(linspace_args) == len(self.sys.state)*3
        
        self.state_grids = []
        for i in range(len(self.sys.state)):
            # discrete grid for state i
            grid_i = np.linspace(*linspace_args[i*3:i*3+3])
            self.state_grids.append(grid_i)
        
        grid_size = 'x'.join([str(len(grid)) for grid in self.state_grids])
        print('state discretized on a {:s} points grid'.format(grid_size))
    # end discretize_state
    
    def discretize_control(self, *linspace_args):
        '''create a regular discrete grid for each control variable'''
        assert len(linspace_args) == len(self.sys.control)*3
        
        self.control_grids = []
        for i in range(len(self.sys.control)):
            # discrete grid for state i
            grid_i = np.linspace(*linspace_args[i*3:i*3+3])
            self.control_grids.append(grid_i)
        self.state_grids = state_grids
        
        grid_size = 'x'.join([str(len(grid)) for grid in self.control_grids])
        print('control discretized on a {:s} points grid'.format(grid_size))
    # end discretize_control
    
    def solve_step(self, x_k, J_next):
        '''solve one DP step at a given state value x_k'''
        return (J_k, u_k)
    
    def estimate_cost(self, x_k, u_k, J_next):
        '''estimate cost J_k, the Expectation of g(x_k, u_k) + J_{k+1}(f(x_k, u_k, w_k))
        for a given state x_k, a given control u_k and a cost-to-go J_{k+1}
        '''
        pass
# end DPSolver


if __name__ == '__main__':
    ### Example usage with a Energy Storage system
    import scipy.stats as stats

    ### Storage dynamics:
    # Storage rated energy and power:
    E_rated = 7.2 # [MWh]
    P_rated = 2 # [MW]
    # storage loss factor
    a = 0.05

    # Storage request AR(1) model parameters:
    P_req_scale = 1.5 # [MW]
    phi = 0.8
    innov_scale = P_req_scale*np.sqrt(1- phi**2)
    innov_law = stats.norm(loc=0, scale=innov_scale)

    def dyn_sto(E, P_req, P_sto, P_cur, innov):
        '''state transition function `f(x_k,u_k,w_k)` of a Energy storage
        returns (E(k+1), P_req(k+1))
        '''
        # 1) Stored energy evolution:
        E_next = E + P_sto - a*abs(P_sto)
        # 2) Storage request AR(1) model:
        P_req_next = phi*P_req + w
        return (E_next, P_req_next)

    
    ### Cost model
    c_dev = 200 # [€/MWh]
    
    def cost_lin(E, P_req, P_sto, P_cur, innov):
        '''cost of one instant (linear penalty on the absolute deviation)'''
        P_dev = P_req - P_cur - P_sto
        return c_dev * np.abs(P_dev)

    ### Create the system description:
    sys = SysDescription((2,2,1), name='NaS Storage')
    sys.dyn = dyn_sto
    sys.cost = cost_lin
    sys.perturb_laws = [innov_law]

    sys.print_summary()
    
    ### Create the DP solver:
    dpsolv = DPSolver(sys)
    # discretize the state space
    N_E = 51
    N_P_req = 41
    dpsolv.discretize_state(0, E_rated, N_E,
                            -4*P_req_scale, 4*P_req_scale, N_P_req)
