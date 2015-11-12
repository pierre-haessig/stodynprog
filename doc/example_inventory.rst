::::::::::::::::::::::
A Step-by-step Example
::::::::::::::::::::::

.. include:: links_names.txt

We illustrate the use of StoDynProg to solve a *stochastic optimal control* problem
(i.e. dynamic optimization) by an example of inventory control.

============================================
Description of the Inventory Control Problem
============================================
A simple discrete time control problem, with a **1D discrete state**.
We borrow the notations of Dimitri Bertsekas in his book `Dynamic Programming and Optimal Control`_ where this problem is thoroughly analyzed.

*The story:* As a shop owner, we have at time :math:`k` a stock :math:`x_k` of a certain item.
We want to satisfy a stochastic demand :math:`w_k`
and for doing so we have the possibility to order :math:`u_k` additional items.

*Goal:* Optimal control aims at finding the optimal ordering policy to minimize
the operational cost of the shop (see :ref:`Inventory-cost-description` below).
The *policy* means that we want to find a *function* that yields the optimal
order for each stock :math:`\mu: x \mapsto u`.

The stock evolves according to:

.. math::
    x_{k+1} = x_k + u_k - w_k

This is the *dynamical equation* of this problem, with :math:`x_k` being
the state variable, :math:`u_k` being the control and :math:`w_k` a perturbation.
Let's implement this with the `SysDescription` class.

Dynamics description
--------------------

>>> from stodynprog import SysDescription
>>> invsys = SysDescription((1,1,1), name='Shop Inventory')

This creates a `SysDescription` object for a problem with 1 state variable,
1 control and 1 perturbation. We now fill the `dyn` property of this object.

>>> def dyn_inv(x, u, w):
>>>     'dynamical equation of the inventory stock `x`. Returns x(k+1).'
>>>     return (x + u - w,) # tuple, to support several state variables
>>> # Attach the dynamical equation to the system description:
>>> invsys.dyn = dyn_inv

To complete the description of the system, we also need a probabilitic description
of the stochastic demand :math:`w_k`

.. math::
    P(w_k = j) = p_j
    

We can build the description of the random variable with `scipy.stats`

>>> import scipy.stats as stats
>>> demand_values = [0,   1,   2,   3]
>>> demand_proba  = [0.2, 0.4, 0.3, 0.1]
>>> demand_law = stats.rv_discrete(values=(demand_values, demand_proba))
>>> demand_law = demand_law.freeze()

We can test that the description of the random variable is working

>>> demand_law.pmf([0, 3]) # Probality Mass Function
array([ 0.2,  0.1])
>>> demand_law.rvs(10) # Random Variables generation
array([3, 0, 2, 1, 3, 3, 1, 0, 1, 3])

Then we can fill the `perturb_laws` parameter of the system description:

>>> invsys.perturb_laws = [demand_law] # a list, to support several perturbations

Finally, we have to describe the admissible control region :math:`U(x_k)`
in which the control :math:`u_k` should live.
For now, StoDynProg supports intervals:
:math:`U(x_k) = [u_{min}(x_k), u_{max}(x_k)]`

>>> def admissible_orders(x):
>>>        'interval of allowed orders U(x_k)'
>>>        U1 = (0, 10)
>>>        return (U1, ) # tuple, to support several controls
>>> # Attach it to the system description.
>>> invsys.control_box = admissible_orders


.. _inventory-cost-description:

Cost description
----------------

The decision to place an order or not, and how much to order is dictated
by the goal to minimize the cost.

At each instant the cost is the sum of a holding/shortage cost and
an ordering cost: :math:`g(x_k, u_k) = r(x_k) + c.u_k`.

For holding/shortage cost, we choose a piecewise linear function:

.. math::
    r(x) = \begin{cases}
                &h.x  \quad  \text{ if } x \geq 0 \\
               -&p.x  \quad  \text{ if } x <0
           \end{cases}

The goal of dynamic optimization is to minimize the *total* cost over :math:`N` periods,
that is the sum of instantaneous costs:

.. math::
    J = E \Big\{ \sum_{k=0}^{N-1} r(x_k) + c u_k \Big\}


Python implementation of the cost function (with h=0.5, p=3, c=1):

>>> (h,p,c) = 0.5, 3, 1
>>> def op_cost(x,u,w):
>>>    'operational cost of the shop'
>>>    holding = x*h
>>>    shortage = -x*p
>>>    order = u*c
>>>    return np.where(x>0, holding, shortage) + order


We can check that the cost function indeed yields expected results.
In particular, it is required that vectorized evaluation is working
(in this example, this is achieved by using `np.where`
instead of if/else statements).

>>> op_cost(1,1,0)
1.5
>>> # Vectorized cost computation capability (required):
>>> import numpy as np
>>> op_cost(np.array([-2,-1,0,1,2]),1,0)
array([ 7. ,  4. ,  1. ,  1.5,  2. ])

Finally, we add the cost function to the system description:

>>> invsys.cost = op_cost


Summary of the system description:

>>> invsys.print_summary()
Dynamical system "Shop Inventory" description
* behavioral properties: stationnary, stochastic
* functions:
  - dynamics    : __main__.dyn_inv
  - cost        : __main__.op_cost
  - control box : __main__.admissible_orders
* variables
  - state        : x (dim 1)
  - control      : u (dim 1)
  - perturbation : w (dim 1)


================================
Solving the Optimization Problem
================================

We now use a `DPSolver` instance to solve the dynamic optimization problem.
The solver receives a system description at initialization to access
its information.

>>> from stodynprog import DPSolver
>>> dpsolv = DPSolver(invsys)

Discretization of the problem
-----------------------------

The DPSolver needs some parameters of the problem on how to discretize
1) the state space, 2) the perturbation and 3) the control.
Since the inventory problem is in fact discrete, there is no real choice
for discretization, except the range of the state :math:`x_k`.


1) Discretize the state.

>>> xmin, xmax = (-3,6)
>>> N_x = xmax-xmin+1 # number of states
>>> dpsolv.discretize_state(xmin, xmax, N_x)
[array([-3., -2., -1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.])]

2) Discretize the perturbation

>>> N_w = len(demand_values)
>>> dpsolv.discretize_perturb(demand_values[0], demand_values[-1], N_w)
([array([ 0.,  1.,  2.,  3.])], [array([ 0.2,  0.4,  0.3,  0.1])])

3) Control discretization: we just need to pass the step, since the range
   is already provided by `invsys.control_box`.

>>> dpsolv.control_steps=(1,)


Summary of the solver:

>>> dpsolv.print_summary()
* state space discretized on a 10 points grid
  - Δx = 1
* perturbation discretized on a 4 points grid
  - Δw = 1
* control discretization steps:
  - Δu = 1
    yields 11 possible values


Running Dynamic Programming
---------------------------

Terminal cost vector: we choose a null cost.

>>> J_0 = np.zeros(N_x)

Run one Value Iteration

>>> J,u = dpsolv.value_iteration(J_0)
value iteration run in 0.00 s
>>> J # cost to go of one step
array([ 9. ,  6. ,  3. ,  0. ,  0.5,  1. ,  1.5,  2. ,  2.5,  3. ])
>>> u[...,0] # optimal order decision
array([ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.])

After one step, the optimal order is :math:`\mu(x) = 0` for all :math:`x`.
Clearly, there is room for improvement.
We run a few more iterations:

>>> J,u = dpsolv.value_iteration(J) # 2 steps
value iteration run in 0.00 s
>>> u[...,0]
array([ 4.  3.  2.  1.  0.  0.  0.  0.  0.  0.])
>>> J,u = dpsolv.value_iteration(J) # 3 steps
value iteration run in 0.00 s
>>> u[...,0]
array([ 5.  4.  3.  2.  1.  0.  0.  0.  0.  0.])
>>> J,u = dpsolv.value_iteration(J) # 4 steps
value iteration run in 0.00 s
>>> u[...,0]
array([ 5.  4.  3.  2.  1.  0.  0.  0.  0.  0.])

Plot of the policy:

>>> from pylab import *
>>> xr = range(xmin, xmax+1)
>>> # or equivalent:
>>> xr = dpsolv.state_grid[0]
>>> plot(xr, u, '-x')
>>> # Annotations:
>>> title('Optimal ordering policy')
>>> xlabel('Stock $x_k$')
>>> ylabel('Number of items to order $u_k$')
>>> ylim(-0.5, u.max()+0.5)

.. plot:: example_inventory_plot_policy.py

After a few value iterations, the optimal policy appears to converge
to a so-called "(s,S)" policy, where there is a *critical stock* level :math:`S`
that should be reached when ordering.

.. math::
    \mu(x) = \begin{cases}
               S - x&  \quad  \text{ if } x < S \\
                   0&  \quad  \text{ if } x \geq S
             \end{cases}

This optimal threshold value depends on the number of periods of the problem.
With can see that after 2 iterations, we have :math:`S=1`, and for 3 iterations or more,
:math:`S=2`.
Of course, the value of this critical stock depends on the distribution of the demand,
as well as the different cost parameters (h,p,c).
A rigorous treament of Inventory Control can be found in
D. Bertsekas' `Dynamic Programming and Optimal Control`_ book (section 4.2).

========================
Simulation of the System
========================
*(simulation of the Inventory Control, with the optimal policy)*

Building the Policy Function
----------------------------

The simulation of the Inventory Control system needs an order policy *function*
instead of just a vector.
This policy being defined on a discrete state, it can be implemented with a simple index search::

    x_grid = dpsolv.state_grid[0]
    def order_pol(x):
        'ordering policy, for a given stock level `x`'
        x_ind = np.where(x_grid==x)[0]
        return u[x_ind]

A simpler alternative is to use an interpolation over state values provided with `DPSolver`.
(This is inevitable with a continous state problem)
 
>>> order_pol = dpsolv.interp_on_state(u[...,0])

Simulation loop
---------------

Number of instants to simulate:

>>> N = 20


Simulation of a trajectory of the discrete time system
can be done with a `for` loop::

    # Initialize variables
    stock = np.zeros(N+1)
    order = np.zeros(N)
    np.random.seed(0)
    demand = demand_law.rvs(N)

    x0 = 0 # Initial stock
    for k in range(N):
        order[k] = order_pol(stock[k])
        stock[k+1] = invsys.dyn(stock[k], order[k], demand[k])[0]

And now plot the `stock`, `order` and `demand` vectors:

.. plot:: example_inventory_plot_simulation.py 


Finally, the fun comes from playing with the different parameters of the problem:

* changing the demand law
* changing the cost parameters:
  holding cost h and shortage cost p.

