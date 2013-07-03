::::::::::::::::::::::::::
Introduction to StoDynProg
::::::::::::::::::::::::::

.. include:: links_names.txt

StoDynProg is a tool to help solving *stochastic optimal control* problems,
also called dynamic optimization.
It implements some algorithms of Dynamic Programming to solve such problems
(Value Iteration, Policy Iteration algorithms) and most importantly provides
an API to describe the optimization problem.

Goals and Scope
===============

This tool was created for research on the energy management
of Energy Storage Systems like batteries (see `Author's website`_ for details).
Even though it was written to have some generality, it's probably somehow specific.
Indeed, StoDynProg design may biaised towards handling features
of author-specific problems.

Let's try to clarify the scope with the enumeration of these features.
StoDynProg is meant to solve problems with:

* *discrete time* (compulsory),
* *continous state space*, which is discretized on a rectangular grid
  (so discrete state space works as well),
* a state space of *small dimension*
  (say less that 4 or 5, due to the "Curse of Dimensionality")
* a control space of small dimension as well (say 1 or 2).

This code is influenced by the book from which I learned Dynamic Optimization:
`Dynamic Programming and Optimal Control`_ by Dimitri Bertsekas.
I recommend it warmly.


StoDynProg API
==============

Within the `stodynprog` modules, two main classes are available.

1. `SysDescription` is there for the problem description. It holds references
   to the state variables description, the dynamical function and the cost function.
2. `DPSolver` provides methods for solving the optimization problem
   (with Value Iteration or Policy Iteration).
   It also holds the parameters of the solving algorithm like the state discretization.
