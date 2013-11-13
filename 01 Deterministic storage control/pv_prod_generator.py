#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Generates a PV production time series.

The PV series has a daily pattern and some noise to make it "look realistic".
However it is, by no means, meant to be *statistically* realistic.

Pierre Haessig — November 2013
"""

from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.stats as stats

output = 'pv_prod.csv'

# Rated PV power:
P_pv = 1.

dt = 1. # [hours]
N = 24*10
t = np.arange(N)*dt

# A naive clear sky production:
T = 24 # [hours]
sine = -np.cos(2*np.pi * t/T)
P_clear = np.where(sine <0, 0, sine)
P_clear *= P_pv

# Mix in some noise:
np.random.seed(0)
white = np.random.normal(size=N)
# correlation between two time steps:
phi = 0.8
ar_noise = sig.lfilter([1], [1, -phi], white)
# map the gaussian variable to uniform on [0,1]
noise_01 = stats.norm.cdf(ar_noise,scale=1/np.sqrt(1-phi**2))

P_prod = P_clear * noise_01

# Plot:
plt.plot(t, P_prod)
plt.plot(t, P_clear, 'r--')
plt.show()

data = np.hstack((t[:,None],P_prod[:,None]))

print('writing {:d} data lines to "{:s}"...'.format(N, output))
np.savetxt(output, data, header='time,P_prod',
           fmt=['%.1f', '%.4f'], delimiter=',', comments='')



