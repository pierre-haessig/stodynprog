#!/usr/bin/python
# -*- coding: utf-8 -*-
""" SEAREV simulation data (time series provided by Thibaut K.):

* load function
* torque_law function

Pierre Haessig — July 2013
"""

from __future__ import division, print_function, unicode_literals
import numpy as np
import os.path


# SEAREV parameters
damp = 4.e6 # N/(rad/s)
torque_max = 2e6 # N.m
power_max = 1.1 # MW

# timestep
dt = 0.1 # [s]

# Sea state
Hs = 3. # [m]
Tp = 9. # [s]


# Searev data path:
data_dir = 'data'


def load(fname='Em_1.txt'):
    '''Load SEAREV data file `fname` ('Em_1.txt', 'Em_2.txt' or 'Em_3.txt')
    
    Returns:
    t, elev, angle, speed, torque, accel
    '''
    assert fname in ['Em_1.txt', 'Em_2.txt', 'Em_3.txt']
    fname = os.path.join(data_dir, fname)
    print('loading SEAREV simulation data "%s"' % os.path.basename(fname))    
    data = np.loadtxt(fname, skiprows=4)

    # split columns:
    t, elev, angle, speed, torque = data.T
    # backward derivative:
    accel = np.diff(speed)/dt 
    accel = np.concatenate(([0.], accel)) 
    #power=speed*torque/1e6 # [MW]

    # Regenerate the time vector because there are some irregularities:
    n_pts = len(speed)
    t = np.arange(n_pts)*dt
    return (t, elev, angle, speed, torque, accel)


### Searev "PTO strategy": torque(speed) and power(speed)


def torque_law(speed):
    '''Torque command law ("PTO strategy") of the SEAREV'''
    tor = speed * damp
    # 1) Max torque limitation:
    tor = np.where(tor >  torque_max,  torque_max, tor)
    tor = np.where(tor < -torque_max, -torque_max, tor)
    # 2) Max power limitation:
    tor = np.where(tor*speed > power_max*1e6, power_max*1e6/speed, tor)
    return tor

def searev_power(speed):
    '''Searev power-take-off as function of speed (rad/s)
    returns P_prod (MW)
    '''
    tor = speed * damp
    # 1) Max torque limitation:
    tor = np.where(tor >  torque_max,  torque_max, tor)
    tor = np.where(tor < -torque_max, -torque_max, tor)
    # 2) Max power limitation:
    P_prod = tor*speed/1e6 # W -> MW
    P_prod = np.where(P_prod > power_max, power_max, P_prod)
    return P_prod

if __name__=='__main__':
    t, elev, angle, speed, torque = load('Em_1.txt')
    power = speed*torque/1e6 # [MW]
    # Print quick statistics:
    print('mean power: %.3f MW' % power.mean())
    print('speed std: %.3f m/s' % speed.std())
