#!/usr/bin/python
# -*- coding: utf-8 -*-
""" fit a time-series model to SEAREV power production data

the model is based on an AR(2) for speed data
then speed is transformed into power knowing the speed->torque function

Pierre Haessig — April 2013
"""

from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.signal as sig

from scipy.optimize import fmin
from scipy.linalg import toeplitz

import statsmodels.api
from statsmodels import tsa

try:
    import armaToolbox
except ImportError:
    import sys
    sys.path.append('../../../30 Data Toolbox')
    import armaToolbox


from searev_data import load, torque_law, damp, dt

### Load time series:
fname = 'Em_1.txt'
t, elev, angle, speed, torque, accel = load(fname)
power = speed*torque/1e6 # [MW]

n_pts = len(speed)


def acov(x, maxlags, lags=None):
    '''auto covariance of x from lag 0 to `maxlags`
    (or at specific `lags` if not None)
    '''
    if lags is None:
        lags = range(maxlags+1)
    N = len(x)
    acov_x = np.zeros(len(lags))
    for h in lags:
        acov_x[h] = np.sum((x[h:]*x[:N-h]))/N
    return acov_x

def acf(x, maxlags, lags=None):
    '''auto covariance of x from lag 0 to `maxlags`
    (or at specific `lags` if not None)
    '''
    acov_x = acov(x, maxlags, lags)
    # Normalize autocovariance at lag 0:
    acf = acov_x/acov_x[0]
    return acf

# TODO : write tests for this function.
# TODO 2: compare this code with arima_process.arma_acovf from statsmodel.tsa
def arma_acov(model, maxlags, innov_var=1.):
    '''true autocovariance of an arma model from lag 0 to `maxlags`
    with innovation variance `innov_var` (default to 1.0)
    
    Reference: Brockwell & Davis formulas (3.3.8) and (3.3.9), p. 93 2nd edition
    '''
    # Grab ARMA parameters
    ar_coef = np.asarray(model['ar'])
    ma_coef = np.asarray(model['ma'])
    # ARMA order:
    p = len(ar_coef)
    q = len(ma_coef)
    
    # Impulse response:
    delta = np.zeros(q+1)
    delta[0] = 1
    num = np.concatenate(([1],ma_coef))
    den = np.concatenate(([1],-np.array(ar_coef)))
    imp_res = sig.lfilter(num, den, delta) # h(0) -- h(q)
    assert len(imp_res) == (q+1)
    
    # Number of covariance steps to solve with linear equations
    n = max(p, q+1) # from cov(0) to cov(n)
    
    imp_res_rev = np.concatenate((imp_res[::-1], np.zeros(n)))
    b = sig.lfilter(num, [1], imp_res_rev)
    b = b[q:]
    assert len(b) == (n+1)
    b = b*innov_var
    
    
    ar_coef_ext = np.zeros(n)
    ar_coef_ext[:p] = ar_coef
    r = np.concatenate((-ar_coef_ext[::-1], [1], np.zeros(n))) # 2n+1 vector
    c = np.zeros(n+1)
    c[0] = -ar_coef_ext[-1]
    M = toeplitz(c,r)
    # Folded addition, due to the symetry cov(-k) = cov(k)
    M_neg, M_0, M_pos = np.hsplit(M, (n,n+1))
    
    A = np.hstack((M_0, M_pos + M_neg[:,::-1]))
    
    # Solve for cov(0) to cov(n)
    cov_n = np.linalg.solve(A,b)
    
    if maxlags <= n:
        return cov_n[:maxlags+1]
    
    # else maxlags > n we need to complete the covariance
    cov = np.zeros(maxlags+1)
    # Paste the first n lags:
    cov[:n+1] = cov_n
    
    # Apply the AR filter formula to find the remaining covariances
    for i in range(n+1, maxlags+1):
        cov[i] = np.inner(cov[i-p:i], ar_coef[::-1])
    
    return cov

def arma_acf(model, maxlags):
    '''true autocorrelation of an arma model from lag 0 to `maxlags`
    '''
    acov = arma_acov(model, maxlags)
    # Normalize autocovariance at lag 0:
    acf = acov/acov[0]
    return acf

def ar_acf_fit(x, p, maxlags):
    '''fit an AR(p) model to x based on the least-square fit
    of the autocorrelation function from lag 0 to `maxlags`.
    
    the variance of the residuals is adjusted so that the variance
    of the data and the process match.
    
    returns (ar_coef, res_var)
    '''
    assert p >= 0
    n = len(x)
    # first guess by one-step AR least-square estimation
    # Regressor matrix:
    X = np.zeros((n-p, p))
    for i in range(p):
        # col `i` contains x lagged by i+1 (from 1 to p)
        X[:,i] = x[i+1:n-p+i+1]
    y = x[:n-p]
    ar_coef0, _,_,_ = np.linalg.lstsq(X,y)
    
    def cov_objective(ar_coef):
        # Compute mean square error of AR residuals
        model = {'ar':ar_coef, 'ma':[]}
        acf_ar = arma_acf(model, maxlags)
        acf_x = acf(x, maxlags)
        acf_error = acf_ar - acf_x
        return np.mean(acf_error**2)
    
    if maxlags is not None:
        # TODO : find a better optimizer ! (pb of local optima)
        ar_coef = fmin(cov_objective, ar_coef0,
                       xtol=1e-6, ftol=1e-6)
    else:
        ar_coef = ar_coef0
    # Compute the variance of residuals so that variance of both processes match:
    res_var = (acov(x, 0)/arma_acov({'ar':ar_coef, 'ma':[]}, 0))[0]
    
    return ar_coef, res_var

# Fit using Conditional maximum likelihood:
sm_ar_mod = tsa.ar_model.AR(speed)
sm_ar_results = sm_ar_mod.fit(2, trend='nc')
sm_ar_coef = sm_ar_results.params
sm_innov_var = sm_ar_results.sigma2
print('Speed AR(2) model from CMLE:')
print(' * AR coef: '+ str(sm_ar_coef))
print(' * innovation sd: {:g}'.format(np.sqrt(sm_innov_var)))

# Fit an AR(2) model to the speed data
ar_fitlags = 150

ar_coef, innov_var = ar_acf_fit(speed, 2, maxlags = ar_fitlags)
print('Speed AR(2) model from acf fit on %.0f s:' % (ar_fitlags*dt))
print(' * AR coef: '+ str(ar_coef))
print(' * innovation sd: {:g}'.format(np.sqrt(innov_var)))



# speed series simulation:
model = {'ar':ar_coef, 'ma':[]}
innov_gen = lambda size: np.random.normal(scale = np.sqrt(innov_var), size=size)
speed_sim = armaToolbox.arima_sim(model, n_pts, innov_gen = innov_gen)
accel_sim = np.zeros(n_pts)
accel_sim[:-1] = np.diff(speed_sim)/dt

power_sim = speed_sim*torque_law(speed_sim)/1e6

### State space stochastic model (VAR kind of) #################################
# state variables:
X = np.vstack((speed, accel))
p1, p2 = ar_coef
# Conversion of AR2 to state-space
T_AR2 = np.array([[p1+p2,     -dt*p2],
                  [(p1+p2-1)/dt, -p2]])
T = T_AR2

fitlags = 200
def pred_objective(T):
    '''mean square prediction error at horizon `fitlags`'''
    T = T.reshape((2,2))
    X_prev = X
    err = 0
    # Recursive prediction:
    for i in range(1, fitlags+1):
        X_prev = np.dot(T,X_prev)
        err_prev = X[:,i:] - X_prev[:,:-i]
        # Add the mean square error:
        err += np.sum(err_prev**2)/(n_pts-i)/fitlags
    return err
# end objective function

print('fitting prediction along {:d} steps'.format(fitlags))
#T = fmin(pred_objective, T_AR2.ravel())

T = T.reshape((2,2))
# Evaluation 
err = pred_objective(T)
print('mean square error of multi-step ss pred: {:g}'.format(err))

# Model the prediction error at one step:
X_prev_1 = np.dot(T, X)
err_prev = X[:,1:] - X_prev_1[:,:-1]
# Covariance of the innnovation:
innov_cov = np.cov(err_prev)

innov_cov = np.array([[1, 1/dt],
                      [1/dt, 1/dt**2]])*innov_var


print('innovation covariance:')
print(innov_cov)

# Compute the prediction and a simulation trajectory:
maxlags = 1000
X_prev = np.zeros((2,maxlags+1))
X_prev[:,0] = X[:,n_pts-1]
X_traj = np.zeros((2,maxlags+1))
X_traj[:,0] = X[:,n_pts-1]

# Simulate some innovation:
rnd = np.random.RandomState()
err_sim = rnd.multivariate_normal([0,0], innov_cov, maxlags).T

for i in range(1, maxlags+1):
    X_prev[:,i] = np.dot(T,X_prev[:,i-1])
    X_traj[:,i] = np.dot(T,X_traj[:,i-1]) + err_sim[:,i-1]
#    reg = torque_law(X_prev[[1],i-1])/damp-X_prev[1,i-1]
#    X_reg = np.concatenate((X_prev[:,i-1], reg))

#    X_prev[:,i] = np.dot(Topt,X_reg)
#    X_traj[:,i] = np.dot(Topt,X_reg) + err_sim[:,i-1]



### Non-linear AR(2) model #####################################################
n_traj = 10
# Starting values of NLAR2 from AR(2) estimation:
ps = p1+p2 # effect of speed
pa = -dt*p2 # effect of acceleration
pt = 0. # effect of torque (non-linear part)
ptv = 0. # effect of torque variations
innov_std = np.sqrt(innov_var)

def NLAR_sim(params, S, S_lag):
    '''simulate 1 step of a non-linear AR(2) model of speed
    
    params = ps, pa, pt, ptv, std
    '''
    ps, pa, pt, ptv, std = params
    A = (S-S_lag)/dt

    Tnl = torque_law(S)/damp - S
    Tnl_lag = torque_law(S_lag)/damp - S_lag
    Tvar = (Tnl - Tnl_lag)/dt

    innov = rnd.normal(scale=innov_std, size=S.shape)
    # dynimical equation:
    S_next = ps*S + pa*A + pt*Tnl + ptv*Tvar + innov
    return S_next

def NLAR_sim_traj(params, S, S_lag, n_sim):
    '''simulate trajectories of length (n_sim+2)
    of a non-linear AR(2) model of spee
    '''
    S = np.atleast_1d(S)
    S_lag = np.atleast_1d(S_lag)
    n_traj = len(S)
    S_sim = np.zeros((n_sim+2, n_traj))
    S_sim[0] = S_lag
    S_sim[1] = S
    for i in range(2,n_sim+2):
        S_sim[i] = NLAR_sim(params, S_sim[i-1], S_sim[i-2])
    return S_sim

def NLAR_pred_error(params, fitlags=100):
    '''quadratic prediction error of the NL AR(2) model'''
    ps, pa, pt, ptv, std = params
    
    # Initialize
    S_traj = np.repeat(np.atleast_2d(speed), n_traj, 0).T
    assert S_traj.shape == (len(speed), n_traj)
    S = S_traj[1:]
    S_lag = S_traj[:-1]

    quad_error = 0.
    print('')
    for i in range(fitlags):
        print('\rhorizon {}/{}'.format(i+1, fitlags), end='')
        # one step further:
        S_next = NLAR_sim((ps, pa, pt, ptv, std), S, S_lag)
        # Compute prediction error and accumulate
        prev_error = speed[2+i:] - S_next[:-1-i].mean(axis=1)
        quad_error += np.mean(prev_error**2)/fitlags
        # Prepare next step
        S_lag = S
        S = S_next
    print()
    return quad_error


## Evaluate the average multi-step prediction error:
#qerr= NLAR_pred_error((ps, pa, pt, ptv, innov_std))
#print(qerr)

# Lauch the NLAR optimization:
param_0 = (ps, pa, pt, ptv, innov_std)
# param_opt100 = fmin(NLAR_pred_error, param_0, maxfun=100)
param_opt100 = np.array([  9.91962946e-01,   9.82452953e-02,
                           3.06547920e-04, 7.02369575e-05,
                           3.48512704e-03]) # precomputed

# Conclusion : NLAR2 is not working well !!

### Plot #######################################################################
mpl.rcParams['grid.color'] = (0.66,0.66,0.66, 0.4)


### 1) Autocovariance:
fig = plt.figure('acf', figsize=(6,3))

ax = fig.add_subplot(111, #title='speed autocorrelation compared with models',
                          xlabel='lag time (s)')
maxlags = 400 # 40 s
t_lags = np.arange(maxlags+1)*dt
ax.plot(t_lags, acf(speed, maxlags), 'b-', lw=2,
        label='data acf')
ax.plot(t_lags, arma_acf({'ar':ar_coef, 'ma':[]}, maxlags), 'g-',
        label='AR(2) model - acf fit on %.0f s' % (ar_fitlags*dt))
ax.plot(t_lags, arma_acf({'ar':sm_ar_coef, 'ma':[]}, maxlags), 'c--',
        label='AR(2) model - cmle fit')
ax.legend(loc='upper right', prop={'size':10}, borderaxespad=0.)

# fine tune the legend:
box = ax.get_legend().get_frame()
box.set_linewidth(0.5) # thin border
box.set_facecolor([1]*3) # white
box.set_alpha(.7)

fig.tight_layout()
#fig.savefig('speed_acf_AR2.pdf')



### 2) Trajectory comparison
fig = plt.figure('traj', figsize=(8,5))
# a) acceleration
ax = fig.add_subplot(311, title='Trajectory comparison: data {} vs. model'.format(fname),
                     ylabel=u'acceleration (rad/s²)')
ax.plot(t, accel, label='data')
ax.plot(t, accel_sim + 2, label='AR(2) derivative')
ax.legend(loc='upper right')
# b) speed
ax = fig.add_subplot(312, ylabel='speed (rad/s)', sharex=ax)
ax.plot(t, speed, label='data')
ax.plot(t, speed_sim + 2., label='AR(2) model')
ax.legend(loc='upper right')
# c) Power
ax = fig.add_subplot(313, xlabel='time (s)', ylabel='power (MW)', sharex=ax)
ax.plot(t, power, label='data')
ax.plot(t, power_sim + 1.5, label='AR(2) model')
ax.legend(loc='upper right')



### State space model plot ###

#plt.figure('pred error')

#plt.title('one step prediction error')
#plt.plot(err_prev[0], err_prev[1], '+')
#plt.plot(err_sim[0], err_sim[1], 'rx')
#plt.xlabel('speed error')
#plt.ylabel('angle error')


fig = plt.figure('SS model')

# Plot the past data:
ax1 = fig.add_subplot(211, title=u'State-space model, x=(speed, accel)', 
                     ylabel='speed (rad/s)')
ax2 = fig.add_subplot(212, xlabel='time (s)', ylabel=u'accel (rad/s²)', sharex=ax1)
ax1.plot(t, X[0], label='speed data')
ax2.plot(t, X[1], label='accel data')

maxlags = len(X_prev[0])
t_prev = np.arange(maxlags)*dt + t[-1]
ax1.plot(t_prev, X_prev[0], label='prediction')
ax1.plot(t_prev, X_traj[0], label='stoch. simul.')
ax1.legend(loc='upper left')

ax2.plot(t_prev, X_prev[1], label='prediction')
ax2.plot(t_prev, X_traj[1], label='stoch. simul.')
ax2.legend(loc='upper left')

ax2.set_xlim(t[-1] - 2*maxlags*dt, t[-1] + maxlags*dt*1.05)


### Spectral analysis ### ######################################################
NFFT = 1024


#Pee, f = mpl.mlab.psd(elev,  Fs = 1/dt, NFFT=NFFT)
Pan, f = mpl.mlab.psd(angle, Fs = 1/dt, NFFT=NFFT)
Psp, f = mpl.mlab.psd(speed, Fs = 1/dt, NFFT=NFFT)
Pac, f = mpl.mlab.psd(accel, Fs = 1/dt, NFFT=NFFT)

# asymptote
f0 = 0.15 # Hz -> 6.7 sec

fig = plt.figure('spectrum')

plt.loglog(f, Pan, 'g-', label='angle')
plt.loglog(f, Psp, 'b-', label='speed')
plt.loglog(f, Pac, 'c-', label='accel.')

## AR(2) spectrum:
nu = f*dt*2*np.pi # frequency in [0, pi]
H_ar2 = 1/(1-p1*np.exp(-1j*nu) - p2*np.exp(-2j*nu))
S_ar2 = np.abs(H_ar2)**2*innov_var*2*dt

plt. loglog(f, S_ar2, 'b-', lw=3, alpha=0.3, label='AR(2) for speed')


# asymptotes
f = np.logspace(-1.1, 0.75)
plt.loglog(f, 1/(f/f0)**4, '--', color='#FF5500', label=r'$1/f^4$')
plt.loglog(f, 1/(f/f0)**6, '--', color='orange', label=r'$1/f^6$')
plt.loglog(f, 1/(f/f0)**8, '--', color='red', label=r'$1/f^8$')

plt.title('SEAREV data spectrums "{}"'.format(fname))
plt.xlabel('frequency (Hz)')


plt.legend(loc='lower left')
fig.tight_layout()



#### Plot NLAR2 results:
n_sim = 500
k = 2098
# 10 trajectories
plt.figure('NLAR2')
plt.plot(t[:k+n_sim], speed[:k+n_sim], 'b')

s_sim = NLAR_sim_traj(param_opt100, [speed[k+1]]*10, speed[k], n_sim)
plt.plot(t[k:k+n_sim+2], s_sim, 'g', alpha=0.5)

s_sim = NLAR_sim_traj((ps, pa, pt, -3e-3, innov_std), [speed[k+1]]*10, speed[k], n_sim)
plt.plot(t[k:k+n_sim+2], s_sim, 'r', alpha=0.5)
plt.legend(('data', 'trajectories'))


plt.show()
