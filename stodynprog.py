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
import itertools
from datetime import datetime


def _zero_cost(*x):
    '''zero cost function g(x), used as default terminal cost'''
    return 0.


def _enforce_sig_len(fun, args, shortname=None):
    ''' Enforces the signature length of `fun` to match `args`
    
    Checks that function `fun` indeed accepts len(`args`) arguments,
    raises ValueError othewise.
    Also `shortname` is used, if provided, in the error message to
    prepend fun.__name__
    '''
    fun_args = inspect.getargspec(fun).args
    if not len(fun_args) == len(args):
        # Build an error message of the kind
        # "dynamics function 'dyn_sto' should accept 3 args (x1, u1, w1), not 4."
        err_msg = ''
        if shortname is not None:
            err_msg += shortname
        err_msg += " '{:s}' ".format(fun.__name__)
        err_msg += 'should accept {:d} args ({:s}), not {:d}.'.format(
                    len(args), ', '.join(args), len(fun_args))
        raise ValueError(err_msg)
    else:
        return True
# end _enforce_sig_len

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
        self._dyn_args = self.state + self.control + self.perturb
        if not self.stationnary:
            # for unstationnary systems, instant `k` must be provided as 1st argument
            self._dyn_args.insert(0, 'k')
        
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
        if _enforce_sig_len(dyn, self._dyn_args, 'dynamics function'):
            self._dyn = dyn
        
        # Read the variable names from the signature of `dyn`
        dyn_args = inspect.getargspec(dyn).args
        # Rewrite the internally stored signature
        self._dyn_args = dyn_args
        # Split the signature between state, control and perturb:
        if not self.stationnary:
            # drop the first argument
            dyn_args = dyn_args[1:]
        self.state = dyn_args[0:len(self.state)]
        dyn_args = dyn_args[len(self.state):] # drop state variables
        self.control = dyn_args[0:len(self.control)]
        dyn_args = dyn_args[len(self.control):]  # drop control variables
        self.perturb = dyn_args[0:len(self.perturb)]
    
    @property
    def control_box(self):
        '''control description function U_k(x_k), expressed as a box (Hyperrectangle)
        which means the admissible control set must be described as a
        Cartesian product of intervals U = [u1_min, u1_max] x [u2_min, u2_max] x ...
        '''
        return self._control_box
    
    @control_box.setter
    def control_box(self, control_box):
        '''sets the control description function'''
        # Check the signature length:
        args =  self.state
        if not self.stationnary:
            args.insert(0, 'k')
        if _enforce_sig_len(control_box, args, 'control description function'):
            self._control_box = control_box
    
    @property
    def cost(self):
        '''cost function g_k(x_k, u_k, w_k)'''
        return self._cost
    
    @cost.setter
    def cost(self, cost):
        '''sets the cost function'''
        # Check the signature length:
        if _enforce_sig_len(cost, self._dyn_args, 'cost function'):
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



from scipy.interpolate import RectBivariateSpline

class RectBivariateSplineBc(RectBivariateSpline):
    '''extended RectBivariateSpline class,
    where spline evaluation works uses input broadcast
    and returns an output with a coherent shape.
    '''
    def __call__(self, x, y):
        '''extended `ev` method, which supports array broadcasting
        '''
        x,y = np.broadcast_arrays(x,y)
        # flatten the inputs after saving their shape:
        shape = x.shape
        x = np.ravel(x)
        y = np.ravel(y)
        # Evaluate the spline and reconstruct the dimension:
        z = self.ev(x,y)
        z = z.reshape(shape)
        return z


class DPSolver(object):
    def __init__(self, sys):
        '''Dynamic Programming solver based on Value Iteration
        '''
        self.sys = sys
        # steps for control discretization
        self.control_steps = (1.,)*len(self.sys.control)
    # end __init__()
    
    def discretize_perturb(self, *linspace_args):
        '''create a regular discrete grid for each perturbation variable
        
        grids are stored in `self.perturb_grids` and can also be set manually
        corresponding probability weights are in `self.perturb_proba`
        '''
        assert len(linspace_args) == len(self.sys.perturb)*3
        
        self.perturb_grids = []
        self.perturb_proba = []
        for i in range(len(self.sys.perturb)):
            # discrete grid for perturbation `i`
            grid_wi = np.linspace(*linspace_args[i*3:i*3+3])
            pdf_wi = self.sys.perturb_laws[i].pdf
            proba_wi = pdf_wi(grid_wi)
            proba_wi /= proba_wi.sum()

            self.perturb_grids.append(grid_wi)
            self.perturb_proba.append(proba_wi)
            
        
        grid_size = 'x'.join([str(len(grid)) for grid in self.perturb_grids])
        # Print a report on discretization:
        print('Perturbation discretized on a {:s} points grid'.format(grid_size))
        for i in range(len(self.sys.perturb)):
            step = self.perturb_grids[i][1] - self.perturb_grids[i][0]
            print('  Δ{:s} = {:g}'.format(self.sys.perturb[i], step))
    # end discretize_perturb()
    
    def discretize_state(self, *linspace_args):
        '''create a regular discrete grid for each state variable
        
        grids are stored in `self.state_grids` and can also be set manually.
        '''
        assert len(linspace_args) == len(self.sys.state)*3
        
        self.state_grids = []
        for i in range(len(self.sys.state)):
            # discrete grid for state `i`
            grid_xi = np.linspace(*linspace_args[i*3:i*3+3])
            self.state_grids.append(grid_xi)
        
        grid_size = 'x'.join([str(len(grid)) for grid in self.state_grids])
        # Print a report on discretization:
        print('State space discretized on a {:s} points grid'.format(grid_size))
        for i in range(len(self.sys.state)):
            step = self.state_grids[i][1] - self.state_grids[i][0]
            print('  Δ{:s} = {:g}'.format(self.sys.state[i], step))
    # end discretize_state()
    
    def interp_on_state(self, A):
        '''returns an interpolating function of matrix A, assuming that A
        is expressed on the state grid `self.state_grids`
        
        the shape of A should be (len(g) for g in self.state_grids)
        '''
        # Check the dimension of A:
        expect_shape = tuple(len(g) for g in self.state_grids)
        if A.shape != expect_shape:
            raise ValueError('array `A` should be of shape {:s}, not {:s}'.format(
                             str(expect_shape), str(A.shape)) )
        
        if len(expect_shape) > 2:
            raise NotImplementedError('interpolation for state dimension > 2'
                                      ' is not yet implemented.')
        if len(expect_shape) == 2:
            x1_grid = self.state_grids[0]
            x2_grid = self.state_grids[1]
            A_interp = RectBivariateSplineBc(x1_grid, x2_grid, A, kx=1, ky=1)
            return A_interp
        else:
            raise NotImplementedError('interpolation for state dimension 1'
                                      ' is not yet implemented.')
    # end interp_on_state()
    
    def control_grids(self, state_k):
        '''returns u1_range, u2_range which is a grid on the box
        of admissible controls using self.control_steps as hints
        '''
        # 1) Evaluate the admissible box:
        intervals = self.sys.control_box(*state_k)
        
        # 2) Build the dicretization grid for each control:
        control_grids = []
        for (u_min, u_max), step in zip(intervals, self.control_steps):
            width = u_max - u_min
            npts = width / step # gives a float
            
            if npts < 0.1:
                # step size is much (10x) thinner than the admissible width,
                # only keep one control point at the interval center :
                u_grid = np.array([(u_min+u_max)/2])
            else:
                # ensure we take enough points so that the actual discretization step
                # is smaller or equal than the `step` hint
                npts = np.ceil(npts) + 1
                u_grid = np.linspace(u_min, u_max, npts)
            control_grids.append(u_grid)
        # end for each control
        return control_grids
    # end control_grids()
    
    def solve_step(self, J_next):
        '''solve one DP step the all state space grid
        '''
        t_start = datetime.now()
        # Iterator over the state grid:
        state_grid = itertools.product(*self.state_grids)
        state_dims = tuple(len(grid) for grid in self.state_grids)
        state_ind = itertools.product(*[range(d) for d in state_dims])
        
        # number of control variables
        nb_control = len(sys.control)
        
        # Initialize the 
        J_k = np.zeros(state_dims)
        u_k = np.zeros(state_dims + (nb_control,) )
        
        # Interpolating function of the cost-to-go
        J_next = self.interp_on_state(J_next)
        
        # Loop over the state grid
        print('starting state loop...', end='')
        for ind_x, x_k in itertools.izip(state_ind, state_grid):
            # compute an allowed control grid
            u_grids = self.control_grids(x_k)
            # Iterate over the control grid
            J_xk_opt = np.inf
            u_xk_opt = None
            for u_xk in itertools.product(*u_grids):
                # TODO : vectorize this grid (at least one dimension)
                # Compute the expected cost of control u_k:
                J = self.estimate_cost(x_k, u_xk, J_next)
                if J < J_xk_opt:
                    J_xk_opt = J
                    u_xk_opt = u_xk
            
            # Save the optimal value:
            J_k[ind_x] = J_xk_opt
            u_k[ind_x] = u_xk_opt
            # Report progress:
#            print('\rstate loop {:.1%}...'.format(np.ravel_multi_index(
#                                                  ind_x,(N_E,N_P_req))/(N_E*N_P_req))
#                                                  , end='')
        print('')
        exec_time = (datetime.now() - t_start).total_seconds()
        print('state loop run in {:.2f} s'.format(exec_time))
        
        return (J_k, u_k)
    # end solve_step
    
    def estimate_cost(self, x_k, u_k, J_next):
        '''estimate cost J_k, the Expectation of g(x_k, u_k) + J_{k+1}(f(x_k, u_k, w_k))
        for a given state x_k, a given control u_k and a cost-to-go J_{k+1}
        '''
        # Perturbation (for now only 1D):
        w_k = self.perturb_grids[0]
        w_proba = self.perturb_proba[0]
        # TODO : implement an nD perturbation
        
        args = x_k + u_k + (w_k,)
        # Compute a grid of next steps:
        x_next = self.sys.dyn(*args)
        # Compute a grid of costs:
        J_k_grid = sys.cost(*args) + J_next(*x_next)
        # Expected (weighted mean) cost:
        J_k = np.inner(J_k_grid, w_proba)
        return J_k
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
        P_req_next = phi*P_req + innov
        return (E_next, P_req_next)


    def admissible_controls(E, P_req):
        '''returns the set of admissible control U(x_k) of an Energy storage
        The two controls are P_sto and P_cur.
        
        Returns the cartesian description of the admissible control space
        (u1_min, u1_max), (u2_min, u2_max)
        '''
        # 1) Constraints on P_sto:
        P_neg = np.max(( -E/(1+a), -P_rated))
        P_pos = np.min(( (E_rated - E)/(1-a), P_rated))
        U1 = (P_neg, P_pos) # [P_req, P_pos]
        # 2) Constraints on the curtailment P_cur
        U2 = (0, np.max((P_req,0)) )
        U2 = (0,0) # disable curtailment
        return (U1, U2)

    
    ### Cost model
    c_dev = 200 # [€/MWh]
    
    def cost_lin(E, P_req, P_sto, P_cur, innov):
        '''cost of one instant (linear penalty on the absolute deviation)'''
        P_dev = P_req - P_cur - P_sto
        return c_dev * np.abs(P_dev)
    
    def cost_quad(E, P_req, P_sto, P_cur, innov):
        '''a simple quadratic cost model
        which penalizes only the commitment deviation P_dev
        '''
        P_dev = P_req - P_cur - P_sto
        return P_dev**2 + P_cur**2

    
    ### Create the system description:
    sys = SysDescription((2,2,1), name='NaS Storage')
    sys.dyn = dyn_sto
    sys.control_box = admissible_controls
    sys.cost = cost_quad
    sys.perturb_laws = [innov_law]

    sys.print_summary()
    
    ### Create the DP solver:
    dpsolv = DPSolver(sys)
    # discretize the state space
    N_E = 51
    N_P_req = 41
    dpsolv.discretize_state(0, E_rated, N_E,
                            -4*P_req_scale, 4*P_req_scale, N_P_req)
    # discretize the perturbation
    N_w = 11
    dpsolv.discretize_perturb(-3*innov_scale, 3*innov_scale, N_w)
    # control discretization step:
    dpsolv.control_steps=(.1,.1) # maximum 41 pts when -2,2 MW are admissible
    
    print('Solving one step...')
    J_N = np.zeros((N_E,N_P_req))
    J, u = dpsolv.solve_step(J_N)
    print('done!')
    
    plt.imshow(u[:,:,0], interpolation='nearest')
    plt.show()
    
