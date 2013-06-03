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
        self._control_box = None
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
        '''summary information about the dynamical system'''
        print('Dynamical system "{}" description'.format(self.name))
        ### 1) general properties
        station = 'stationnary' if self.stationnary else 'time dependent'
        stoch = 'stochastic' if self.stochastic else 'deterministic'
        print('* behavioral properties: {}, {}'.format(station, stoch))
        
        ### 2) info about functions:
        print('* functions:')
        funclist = [('dynamics', self.dyn),
                    ('cost', self.cost),
                    ('control box', self.control_box)]
        maxlen = max([len(name) for name, _ in funclist])
        for name, fun in funclist:
            if fun is not None:
                fname = '{0.__module__}.{0.__name__}'.format(fun)
            else:
                fname = 'None (to be defined)'
            print('  - {0:{width}}: {1}'.format(name, fname, width=maxlen+1))
        # end for each function
        
        ### 1) information about variables
        print('* variables')
        vectlist = [('state', self.state),
                   ('control', self.control)]
        if self.stochastic:
            vectlist.append(('perturbation', self.perturb))
        maxlen = max([len(name) for name, _ in vectlist])
        for name, vect in vectlist:
            print('  - {0:{width}}: {1} (dim {2:d})'.format(
                        name,  ', '.join(vect), len(vect), width=maxlen+1 ))
        # end for each vector
    # end print_summary()
    
    def __repr__(self):
        return '<SysDescription "{:s}" at 0x{:x}>'.format(self.name, id(self))
    # end __repr__()
# end SysDescription class

################################################################################
# Interpolation class
# TODO : use a nicer n-dim method (like multilinear interpolation)
from scipy.interpolate import RectBivariateSpline

class RectBivariateSplineBc(RectBivariateSpline):
    '''extended RectBivariateSpline class,
    where spline evaluation works uses input broadcast
    and returns an output with a coherent shape.
    '''
    #@profile
    def __call__(self, x, y):
        '''extended `ev` method, which supports array broadcasting
        '''
        if x.shape != y.shape:
            x,y = np.broadcast_arrays(x,y) # costs about 30µs/call
        # flatten the inputs after saving their shape:
        shape = x.shape
        x = np.ravel(x)
        y = np.ravel(y)
        # Evaluate the spline and reconstruct the dimension:
        z = self.ev(x,y)
        z = z.reshape(shape)
        return z
    # end __call__()
# end RectBivariateSplineBc class

################################################################################
# Stochastic Dynamic Programming class

class DPSolver(object):
    def __init__(self, sys):
        '''Dynamic Programming solver based on Value Iteration
        '''
        self.sys = sys
        # Initialization of discrete grids:
        self.state_grid = [[0.] for s in self.sys.state]
        self.perturb_grid = [[0.] for p in self.sys.perturb]
        self.perturb_proba = [[1.] for p in self.sys.perturb]
        # steps for control discretization
        self.control_steps = (1.,)*len(self.sys.control)
    # end __init__()
    
    def discretize_perturb(self, *linspace_args):
        '''create a regular discrete grid for each perturbation variable
        
        grids are stored in `self.perturb_grid` and can also be set manually
        corresponding probability weights are in `self.perturb_proba`
        '''
        assert len(linspace_args) == len(self.sys.perturb)*3
        
        self.perturb_grid = []
        self.perturb_proba = []
        for i in range(len(self.sys.perturb)):
            # discrete grid for perturbation `i`
            grid_wi = np.linspace(*linspace_args[i*3:i*3+3])
            pdf_wi = self.sys.perturb_laws[i].pdf
            proba_wi = pdf_wi(grid_wi)
            proba_wi /= proba_wi.sum()

            self.perturb_grid.append(grid_wi)
            self.perturb_proba.append(proba_wi)
            
        return self.perturb_grid, self.perturb_proba
    # end discretize_perturb()
    
    def discretize_state(self, *linspace_args):
        '''create a regular discrete grid for each state variable
        
        grids are stored in `self.state_grid` and can also be set manually.
        '''
        assert len(linspace_args) == len(self.sys.state)*3
        
        self.state_grid = []
        for i in range(len(self.sys.state)):
            # discrete grid for state `i`
            grid_xi = np.linspace(*linspace_args[i*3:i*3+3])
            self.state_grid.append(grid_xi)
        
        return self.state_grid
    # end discretize_state()
    
    def interp_on_state(self, A):
        '''returns an interpolating function of matrix A, assuming that A
        is expressed on the state grid `self.state_grid`
        
        the shape of A should be (len(g) for g in self.state_grid)
        '''
        # Check the dimension of A:
        expect_shape = tuple(len(g) for g in self.state_grid)
        if A.shape != expect_shape:
            raise ValueError('array `A` should be of shape {:s}, not {:s}'.format(
                             str(expect_shape), str(A.shape)) )
        
        if len(expect_shape) > 2:
            raise NotImplementedError('interpolation for state dimension > 2'
                                      ' is not yet implemented.')
        if len(expect_shape) == 2:
            x1_grid = self.state_grid[0]
            x2_grid = self.state_grid[1]
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
        control_dims = []
        for (u_min, u_max), step in zip(intervals, self.control_steps):
            width = u_max - u_min
            n_interv = width / step # gives the number of intervals (float)
            
            if n_interv < 0.1:
                # step size is much (10x) thinner than the admissible width,
                # only keep one control point at the interval center :
                npts = 1
                u_grid = np.array([(u_min+u_max)/2])
            else:
                # ensure we take enough points so that the actual discretization step
                # is smaller or equal than the `step` hint
                npts = int(np.ceil(n_interv) + 1)
                u_grid = np.linspace(u_min, u_max, npts)
            control_grids.append(u_grid)
            control_dims.append(npts)
        # end for each control
        return control_grids, tuple(control_dims)
    # end control_grids()
    
    #@profile
    def solve_step(self, J_next, report_time=True):
        '''solve one DP step on the entire state space grid,
        given and cost-to-go array `J_next` discretized over the state space grid
        '''
        t_start = datetime.now()
        # Iterator over the state grid:
        state_grid = itertools.product(*self.state_grid)
        state_dims = tuple(len(grid) for grid in self.state_grid)
        state_ind = itertools.product(*[range(d) for d in state_dims])
        
        # number of control variables
        nb_control = len(self.sys.control)
        
        # Initialize the output arrays
        J_k = np.zeros(state_dims)
        u_k = np.zeros(state_dims + (nb_control,) )
        
        # Interpolating function of the cost-to-go
        J_next_interp = self.interp_on_state(J_next)
        
        # Loop over the state grid
        if report_time: print('starting state loop...', end='')
        for ind_x, x_k in itertools.izip(state_ind, state_grid):
            J_xk_opt, u_xk_opt = self.solve_state_point_vect(x_k, J_next_interp)
            # Save the optimal value:
            J_k[ind_x] = J_xk_opt
            u_k[ind_x] = u_xk_opt
            # Report progress:
#            print('\rstate loop {:.1%}...'.format(
#                  np.ravel_multi_index(ind_x, state_dims) / np.product(state_dims) ),
#                  end='')
        # end for each state value
        exec_time = (datetime.now() - t_start).total_seconds()
        if report_time: print('\rstate loop run in {:.2f} s'.format(exec_time))
        
        return (J_k, u_k)
    # end solve_step
    
    #@profile
    def solve_state_point_loop(self, x_k, J_next_interp):
        '''find the optimal cost J_k and optimal control u_k
        at a given state point `x_k`
        
        This is the *iterative* implentation:
        The set of allowed controls is discretized their expected cost 
        J(x_k, u_k) is computed one after another in a loop.
        Best control and cost is memorized within the loop.
        
        Returns (J_xk_opt, u_xk_opt)
        '''
        # compute an allowed control grid (depends on the state)
        u_grids, control_dims = self.control_grids(x_k)
        # Iterate over the control grid
        J_xk_opt = np.inf
        u_xk_opt = None
        
        # grab the 1D perturbation vector
        w_k = self.perturb_grid[0]
        w_proba = self.perturb_proba[0]
        # TODO : implement an nD perturbation
        
        # Iterate over all possible controls:
        for u_xk in itertools.product(*u_grids):
            ### Compute the expected cost of control u_xk ###
            args = x_k + u_xk + (w_k,)
            # Compute a grid of next steps:
            x_next = self.sys.dyn(*args)
            # Compute a grid of costs:
            J_k_grid = self.sys.cost(*args) # instant cost
            J_k_grid += J_next_interp(*x_next) # add the cost-to-go
            # Expected (weighted mean) cost:
            J = np.inner(J_k_grid, w_proba)
            
            # Check optimality of the cost:
            if J < J_xk_opt:
                J_xk_opt = J
                u_xk_opt = u_xk
        # end for each control
        
        return (J_xk_opt, u_xk_opt)
    # end solve_state_point_loop()
    
    #@profile
    def solve_state_point_vect(self, x_k, J_next_interp):
        '''find the optimal cost J_k and optimal control u_k
        at a given state point `x_k`
        
        This is the *vectorized* implementation:
        The set of allowed controls is discretized and their expected cost
        J(x_k, u_k) is computed *all at once*.
        Then, the best control and cost is found using `np.argmin`
        
        Returns (J_xk_opt, u_xk_opt)
        '''
        # Compute the allowed control grid (depends on the state)
        u_grids, control_dims = self.control_grids(x_k)
        nb_control = len(u_grids)
        
        # Reshape the control grids to enable broadcasted operations:
        for i in range(nb_control):
            # create a tuple of ones of length (nb_control + 1) with -1 at index i
            # (+1 used for the perturbation)
            shape = (1,)*i + (-1,) + (1,)*(nb_control-i)
            # inplace reshape:
            u_grids[i].shape = shape
        
        # grab the 1D perturbation vector
        w_k = self.perturb_grid[0]
        w_proba = self.perturb_proba[0]
        # TODO : implement nD perturbation
        
        args = x_k + tuple(u_grids) + (w_k,)
        # Compute a grid of next steps:
        x_next = self.sys.dyn(*args)
        # Compute a grid of costs:
        g_k_grid = self.sys.cost(*args) # instant cost
        J_k_grid = g_k_grid + J_next_interp(*x_next) # add the cost-to-go
        # Expected (weighted mean) cost:
        J = np.inner(J_k_grid, w_proba) # shape dim_control
        assert J.shape == control_dims
        
        # Find the lowest cost in array J:
        ind_opt = np.unravel_index(J.argmin(),control_dims)
        
        J_xk_opt = J[ind_opt]
        u_xk_opt = [u_grids[i].flatten()[ind_opt[i]] for i in range(nb_control)]
        return (J_xk_opt, u_xk_opt)
    # end solve_state_point_vect()
    
    def print_summary(self):
        '''summary information about the state of the SDP solver
        '''
        print('SDP solver for system "{}"'.format(self.sys.name))
        ### Print a report on Discretization:
        # a) State discretization:
        grid_size = 'x'.join([str(len(grid)) for grid in self.state_grid])
        print('* state space discretized on a {:s} points grid'.format(grid_size))
        for i, grid in enumerate(self.state_grid):
            if len(grid) > 1:
                step = grid[1] - grid[0]
                print('  - Δ{:s} = {:g}'.format(self.sys.state[i], step))
            else: # len(grid) == 1
                print('  - {:s} fixed at {:g}'.format(self.sys.state[i], grid[0]))
        
        # b) Perturbation discretization:
        grid_size = 'x'.join([str(len(grid)) for grid in self.perturb_grid])
        print('* perturbation discretized on a {:s} points grid'.format(grid_size))
        for i, grid in enumerate(self.perturb_grid):
            if len(grid) > 1:
                step = grid[1] - grid[0]
                print('  - Δ{:s} = {:g}'.format(self.sys.perturb[i], step))
            else: # len(grid) == 1
                print('  - {:s} fixed at {:g}'.format(self.sys.perturb[i], grid[0]))
        
        # c) Control discretization
        # Compute the average number of control points:
        control_dims_list = []
        if self.sys.control_box is not None:
            for x_k in itertools.product(*self.state_grid):
                # Compute the control grid dimension for each state
                _, control_dims = self.control_grids(x_k)
                control_dims_list.append(control_dims)
            # Convert list to 2D array for easy stats:
            cdim = np.array(control_dims_list)
        
        print('* control discretization steps:')
        for i in range(len(self.sys.control)):
            step = self.control_steps[i]
            print('  - Δ{:s} = {:g}'.format(self.sys.control[i], step))
            if control_dims_list:
                print('    yields [{:,d} to {:,d}] points ({:,.1f} on average)'.format(
                       cdim[:,i].min(), cdim[:,i].max(), cdim[:,i].mean()) )
        # end for each control
        if control_dims_list:
            cdim_tot = np.prod(cdim, axis=1)
            print('  total: [{:,d} to {:,d}] points ({:,.1f} on average)'.format(
                   cdim_tot.min(), cdim_tot.max(), cdim_tot.mean()) )
        else:
            print('Warning: sys.control_box is still to be defined!')
    # end print_summary()
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
        E_next = E + P_sto - a*abs(P_sto)# + 0*innov
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
    print('')
    
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
    
    dpsolv.print_summary()
    print('')
    
    print('Solving one step...')
    J_N = np.zeros((N_E,N_P_req))
    J, u = dpsolv.solve_step(J_N)
    print('Solving second step...')
    J, u = dpsolv.solve_step(J)
    
    # Make a quick plot of the optimal controls
    fig = plt.figure('optimal controls', figsize=(10,4.5))
    ax1 = fig.add_subplot(121, title='Stored power $P_{sto}$',
                          xlabel=sys.state[1], ylabel=sys.state[0])
    im = ax1.imshow(u[:,:,0], interpolation='nearest')
    fig.colorbar(im)
    ax2 = fig.add_subplot(122, title='Curtailed power $P_{cur}$',
                          xlabel=sys.state[1], ylabel=sys.state[0])
    im = ax2.imshow(u[:,:,1], interpolation='nearest')
    fig.colorbar(im)
    plt.show()
