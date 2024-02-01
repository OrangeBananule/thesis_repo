# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 22:58:22 2020

@author: Cerx
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
from numpy.fft import rfft, irfft, rfftfreq

print('Time start:')
print(time.asctime())


seed = 444
np.random.seed(seed)


from params3d import dt


dti_net = np.load('dti_net_%s.npy'%seed)
dti_x = dti_net[:,0]
dti_y = dti_net[:,1]
dti_z = dti_net[:,2]
t = np.linspace(0,int(dti_z.shape[0])*dt,int(dti_z.shape[0]))

fig,(ax1,ax2) = plt.subplots(1,2)
fig.set_size_inches(8,4)
fig.tight_layout(pad=2)

ax1.plot(t,dti_x,label=r'$d_t I_x$')
ax1.plot(t,dti_y,label=r'$d_t I_y$')
ax1.plot(t,dti_z,label=r'$d_t I_z$')
xlocs = np.linspace(t[0],t[-1],7)
ax1.set_xticks(xlocs)
xxlocs = np.copy(xlocs)
for i in range(xxlocs.shape[0]):
    xxlocs[i] = np.round(xxlocs[i]*1e12,2)
ax1.set_xticklabels(xxlocs)
ax1.set_xlabel(r't ($ps$)')
ax1.set_ylabel(r'$d_t I$ $(C s^{-2})$')
ax1.legend()

ftjz = rfft(dti_z)
w = rfftfreq(dti_z.shape[0],dt)
# ftjz[0] = 0
ax2.plot(w[:100],np.abs(ftjz[:100])/np.amax(np.abs(ftjz[:100])))
xlocs = np.linspace(w[0],w[-1],6)
ax2.set_xticks(xlocs)
ax2.set_xticklabels(np.round(xlocs*1e-12,2))
ax2.set_xlabel(r'$\omega$ ($THz$)')
ax2.set_ylabel(r'FT{$d_t I_z$}')

filename = os.path.basename(__file__)
# file name without extension
filename = os.path.splitext(filename)[0]
fig.savefig('%s_%a.png'%(filename,seed),bbox_inches = 'tight')

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
fig.savefig('%s_smooth_%a.png'%(filename,seed),bbox_inches = 'tight')

fig,ax3 = plt.subplots(1,1)
fig.set_size_inches(4,4)
fig.tight_layout(pad=2)

print(time.asctime())