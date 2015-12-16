#!/usr/bin/python
# -*- coding: utf-8 -*-
"""  Stochastic Dynamic Programming library

Implements naive methods of Dynamic Programming (Value Iteration)
to solve *simple* Optimal Stochastic Control problems

classes : SysDescription, DPSolver

Pierre Haessig — November 2013
"""
from __future__ import (division, print_function, unicode_literals,
                        absolute_import)

from .stodynprog import SysDescription, DPSolver

from stodynprog import tests