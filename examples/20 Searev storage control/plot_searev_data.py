#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Plot Searev time series provided by Thibaut K.

1) plot all time series

2) plot only speed and power, along with an histogram
   (for publication)

Pierre Haessig — April 2013
"""

from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from searev_data import load, damp, torque_law

### Load time series:
fname = 'Em_1.txt'
t, elev, angle, speed, torque, accel = load(fname)
power = speed*torque/1e6 # [MW]

# Print quick statistics:
print('mean power: %.3f MW' % power.mean())
print('speed std: %.3f m/s' % speed.std())

### Plot #######################################################################
mpl.rcParams['grid.color'] = (0.66,0.66,0.66)
mpl.rcParams['grid.alpha'] = 0.4
mpl.rcParams['font.size'] = 10
mpl.rcParams['savefig.dpi'] = 150


fill_alpha = 0.2

### 1) Time series
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True, num='time series')

ax1.set_title('SEAREV time-series "{}"'.format(fname))
ax1.fill_between(t, elev, color='b', alpha=fill_alpha, lw=0)
ax1.plot(t, elev)
ax1.set_ylabel('elevation (m)')

ax2.fill_between(t, angle, color='g', alpha=fill_alpha, lw=0)
ax2.plot(t, angle, color='green')
ax2.set_ylabel('angle (rad)')

ax3.fill_between(t, speed, color='b', alpha=fill_alpha, lw=0)
ax3.plot(t, speed, color='blue', label='speed')
ax3.set_ylabel('speed (rad/s)')
ax3.plot(t, torque/damp, color='red', label=u'torque/damp')
ax3.legend(loc='upper right')

ax4.fill_between(t, accel, color='c', alpha=fill_alpha, lw=0)
ax4.plot(t, accel, color='c')
ax4.set_ylabel(u'accel (rad/s²)')

ax5.plot(t, power)
ax5.set_ylabel('power (MW)')
ax5.set_xlabel('time (s)')


fig.tight_layout()



### 2) Speed/Power time-series with histogram:
fig = plt.figure('speed and power', figsize=(10,4))
fig.suptitle('Speed & Power from a 1000 s simulation of the Searev')
fig.subplots_adjust(bottom=0.09, top=0.99)

from matplotlib.gridspec import GridSpec
gs_time = GridSpec(2, 1)
gs_time.update( left=0.07, right=0.50, hspace=0.05)
gs_timez = GridSpec(2, 1)
gs_timez.update(left=0.51, right=0.84, hspace=0.05)
gs_hist = GridSpec(2, 1)
gs_hist.update( left=0.85, right=0.95, hspace=0.05)


# 1. Speed
ax1t = fig.add_subplot(gs_time[0],
                       ylabel='speed (m/s)')
ax1t.plot(t, speed, color='b')
ax1t.label_outer()

# zoom:
ax1tz = fig.add_subplot(gs_timez[0], sharey=ax1t)
ax1tz.plot(t, speed, color='b')
ax1tz.label_outer()
plt.setp(ax1tz.yaxis.get_ticklabels(), visible=False)
# speed histogram
ax1h = fig.add_subplot(gs_hist[0], sharey=ax1t)
ax1h.hist(speed, bins=30, edgecolor='b', facecolor='#AAAAFF',
          orientation='horizontal',  histtype='stepfilled')
ax1h.set_xticks([])
ax1h.yaxis.tick_right()
#ax1h.set_xlim(0, ax1h.get_xlim()[1]*1.05)

ax1t.set_ylim(-1.1, 1.1)

# 2. Power
ax2t = fig.add_subplot(gs_time[1], sharex=ax1t, 
                       xlabel='time (s)', ylabel='power (MW)')
ax2t.plot(t, power, color='#df025f')

# zoom
ax2tz = fig.add_subplot(gs_timez[1], sharex=ax1tz, sharey=ax2t,
                        xlabel='zoomed time (s)')
ax2tz.plot(t, power, color='#df025f')

plt.setp(ax2tz.yaxis.get_ticklabels(), visible=False)
# power histogram
ax2h = fig.add_subplot(gs_hist[1], sharey=ax2t)
ax2h.hist(power, bins=30, normed=True, edgecolor='#df025f', facecolor='#ffbfda',
          orientation='horizontal',  histtype='stepfilled')
ax2h.set_xticks([])
ax2h.yaxis.tick_right()
ax2h.set_xlim(0, ax2h.get_xlim()[1]*1.05)

ax2t.set_ylim(0, 1.2)

# set the zoomed region
tmin, tmax = 230,300 # (70 seconds)
ax2tz.set_xlim(tmin, tmax)

# Highlight the zoomed region in the unzoomed plot:
recSpeed = mpl.patches.Rectangle((tmin,-2), tmax-tmin, 4,
                                 ls='solid', lw=0.5,
                                 edgecolor=(0.5,)*3, facecolor=(0.9,)*3)
    
ax1t.add_patch(recSpeed)
recPower = mpl.patches.Rectangle((tmin,-1), tmax-tmin, 3,
                                 ls='solid', lw=0.5,
                                 edgecolor=(0.5,)*3, facecolor=(0.9,)*3)
    
ax2t.add_patch(recPower)

# Remove extremal ticks:
ax1t.yaxis.get_major_locator().set_params(prune='both') # extremal speed ticks
ax2t.yaxis.get_major_locator().set_params(prune='both') # extremal power ticks
ax2t.xaxis.get_major_locator().set_params(prune='both') # extremal time ticks
ax2tz.xaxis.get_major_locator().set_params(prune='both') # extremal time ticks


### Phase portrait
fig = plt.figure('phase portrait')

ax = fig.add_subplot(111, title=u'Phase portrait (Ω,a) of the SEAREV\n'
                                 'with leveled viscous torque PTO', 
                          xlabel='speed (rad/s)', ylabel=u'accel (rad/s²)')

s = np.linspace(-1,1, 500)
ax.plot(s, torque_law(s)/damp, 'r', label=u'T(Ω)/damp')
ax.plot(s, s, 'r--', label=u'Ω')
ax.plot(speed, accel, '-', lw=0.3)
ax.legend(loc='upper right')

# 3D phase portrait:
#mlab.plot3d(angle, speed, accel, tube_radius=None, opacity=0.3, color=(0,0,1))

# 3D auto-regression plot:
#from mayavi import mlab
#mlab.points3d(angle[:-1], speed[:-1], speed[1:], mode='point')





plt.show()
