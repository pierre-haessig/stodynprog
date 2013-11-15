StoDynProg
==========

StoDynProg is a Python tool to help solving *stochastic optimal control*
problems, also called *dynamic optimization*.

It implements some algorithms of Dynamic Programming to solve such problems
(Value Iteration, Policy Iteration algorithms)
and most importantly provides an API to describe the optimization problem.
(see `doc/` for more details and examples)

Source code
-----------

Source code is available at https://github.com/pierre-haessig/stodynprog

The source tree contains interpolation code from the dolo project by Pablo Winant
(https://github.com/albop/dolo).

Usage
-----

First a few Cython files must be compiled.
For testing, it's easier to build them *"inplace"*:

    $ make inplace


Then, the path to the code repositery can be added to the `$PYTHONPATH` variable.
For examples, the following lines can be added to the `~/.bashrc` file (for Linux users).

```bash
# Add stodynprog to the PYTHONPATH
export PYTHONPATH='/path/to/stodynprog/git-repositery':$PYTHONPATH
```

To see if the code is properly running, run the tests (requires `nose`):

    $ make test
