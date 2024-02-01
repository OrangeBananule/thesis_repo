# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 22:58:22 2020

@author: Cerx
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, epsilon_0, h, m_e, Boltzmann
from matplotlib.colors import ListedColormap
import time
import sys
import os
from numpy.fft import fft, ifft, fftfreq, rfft, irfft, rfftfreq

print('Time start:')
print(time.asctime())


from params3d import \
    n, epsilon, dt,\
        tR, xwidth, ywidth, zmaterialwidth, \
            xlen, ylen, zlen, \
                zmid, dx, dy, dzmaterial, x, y, zmaterial, \
                    Num_e, Num_h, phi_s, kbT,\
                        me, mh, ve, vh, ve_therm


seeds = [111,222,333,444,555,666,777,888,999,101010]
dti_net = 0
for g in seeds:
    dti_net = dti_net + np.load('dti_net_%s.npy'%g)
dti_net = dti_net/len(seeds)
np.save('dti_avg.npy',dti_net)
# dti_net = np.load('dti_net.npy')
dti_x = dti_net[:,0]
dti_y = dti_net[:,1]
dti_z = dti_net[:,2]
t = np.linspace(0,int(dti_z.shape[0])*dt,int(dti_z.shape[0]))

fig,(ax1,ax2) = plt.subplots(1,2)
fig.set_size_inches(8,4)
fig.tight_layout(pad=2)

# ax1.plot(t,dti_x,label=r'$d_t I_x$')
# ax1.plot(t,dti_y,label=r'$d_t I_y$')
# ax1.plot(t,dti_z,label=r'$d_t I_z$')
ax1.plot(t,dti_x,label=r'$E_x$')
ax1.plot(t,dti_y,label=r'$E_y$')
ax1.plot(t,dti_z,label=r'$E_z$')
xlocs = np.linspace(t[0],t[-1],7)
ax1.set_xticks(xlocs)
xxlocs = np.copy(xlocs)
for i in range(xxlocs.shape[0]):
    xxlocs[i] = np.round(xxlocs[i]*1e12,2)
ax1.set_xticklabels(xxlocs)
ax1.set_xlabel(r't ($ps$)')
# ax1.set_ylabel(r'$\sum_a q_a d_t v_a$ $(C m s^{-1})$')
ax1.set_ylabel(r'E $(N/C)$')
ax1.legend(loc='lower left')
ax1.text(2.75e-12,np.amax(dti_z)*0.92,"(a)",fontsize=18)


ftjz = rfft(dti_z)
w = rfftfreq(dti_z.shape[0],dt)
ax2.plot(w[:100],np.abs(ftjz[:100])/np.amax(np.abs(ftjz[:100])))
xlocs = np.linspace(w[0],w[100],6)
ax2.set_xticks(xlocs)
ax2.set_xticklabels(np.round(xlocs*1e-12,2))
ax2.set_xlabel(r'$\omega (10^{12}Hz$)')
ax2.set_ylabel(r'FT{$E_z$}')
# ax2.set_ylabel(r'FT{$d_t I_z$}')
ax2.text(30.3e12,0.97,"(b)",fontsize=18)

fig.savefig('dti_avg.png',bbox_inches = 'tight')

fig,ax3 = plt.subplots(1,1)
fig.set_size_inches(4,4)
fig.tight_layout(pad=2)


ax3.plot(t[:-1],dti_z[:-1],label='original', color='green')
ftjz[9:] = 0
dti_z = irfft(ftjz)
ax3.plot(t[:-1],dti_z,label='smoothened',linestyle='--', color='red')
xlocs = np.linspace(t[0],t[-1],7)
ax3.set_xticks(xlocs)
xxlocs = np.copy(xlocs)
for i in range(xxlocs.shape[0]):
    xxlocs[i] = np.round(xxlocs[i]*1e12,2)
ax3.set_xticklabels(xxlocs)
ax3.set_xlabel(r't ($ps$)')
ax3.set_ylabel(r'$d_t I$ $(C s^{-2})$')
ax3.legend()

filename = os.path.basename(__file__)
# file name without extension
filename = os.path.splitext(filename)[0]
fig.savefig('dti_avg_smooth.png',bbox_inches = 'tight')

fig,ax3 = plt.subplots(1,1)
fig.set_size_inches(4,4)
fig.tight_layout(pad=2)

print(time.asctime())