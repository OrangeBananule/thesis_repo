# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 22:58:22 2020

@author: Cerx
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os

from params3d import n
from funcs3d import  E_thz


print('Time start:')
print(time.asctime())



fig, (ax1,ax2) = plt.subplots(1,2,subplot_kw={'projection': 'polar'})
fig.set_size_inches(10,4)
fig.tight_layout(pad=2)


dti_net = np.load('dti_avg.npy')
theta_i = np.linspace(-np.arcsin(1/n),np.arcsin(1/n),200)
theta_e = np.arcsin(n*np.sin(theta_i))
counter = 0
counter_end = dti_net.shape[0]
# sys.exit()

while True:
    print(counter)
    if counter == counter_end:
        break
    dti_temp = dti_net[counter]
    if counter%(10) == 0 and counter <=  800:
        #plot E_TE, E_TM
        E_TE,E_TM,E_TMsin,E_TMcos = E_thz(dti_temp,theta_i,theta_e)
        prefac = np.cos(theta_e)/(n*np.cos(theta_i))
        frame1, = ax1.plot(theta_e, prefac*E_TE**2,color='dimgrey')
        frame2, = ax2.plot(theta_e, prefac*E_TM**2,color='dimgrey')
        ax1.set_theta_zero_location("N")
        ax1.grid(True)
        ax1.set_title(r"$|E_{TE}|^2$ transmitted", va='bottom')
        ax1.set_ylim(0,15.5*6e3)
        ylocs = np.linspace(0,15,6)*6e3
        ax1.set_yticks(ylocs)
        ax1.set_yticklabels(ylocs/6e3)
        ax1.text(np.pi*(1.125),(15 + 7.04)*6e3,'*scale by 6e3')
        ax1.plot([np.pi/2,np.pi*3/2],\
                 np.array([1,1])*15*6e3,c='black') #surface
        
        ax2.set_theta_zero_location("N")
        ax2.grid(True)
        ax2.set_title(r"$|E_{TM}|^2$ transmitted", va='bottom')
        ax2.set_ylim(0,15.5*12e3)
        ylocs = np.linspace(0,15,6)*12e3
        ax2.set_yticks(ylocs)
        ax2.set_yticklabels(ylocs/12e3)
        ax2.text(np.pi*(1.125),(15 + 7.04)*12e3,'*scale by 12e3')
        ax2.plot([np.pi/2,np.pi*3/2],\
                 np.array([1,1])*15*12e3,c='black') #surface
        
        
        
        fig.savefig('thz_6_0_5 _ %a.png'%(counter),bbox_inches = 'tight')
        ax1.cla()
        ax2.cla()
        
    elif counter%(10) == 0 and counter > 800:
        #plot E_TE, E_TM
        E_TE,E_TM,E_TMsin,E_TMcos = E_thz(dti_temp,theta_i,theta_e)
        prefac = np.cos(theta_e)/(n*np.cos(theta_i))
        frame1, = ax1.plot(theta_e, prefac*E_TE**2,color='dimgrey')
        frame2, = ax2.plot(theta_e, prefac*E_TM**2,color='dimgrey')
        
        ax1.set_theta_zero_location("N")
        ax1.grid(True)
        ax1.set_title(r"$|E_{TE}|^2$ transmitted", va='bottom')
        ax1.set_ylim(0,30.5*6e3)
        ylocs = np.linspace(0,30,6)*6e3
        ax1.set_yticks(ylocs)
        ax1.set_yticklabels(ylocs/6e3)
        ax1.text(np.pi*(1.125),(30 + 7.04)*6e3,'*scale by 6e3')
        ax1.plot([np.pi/2,np.pi*3/2],\
                 np.array([1,1])*30*6e3,c='black') #surface
        
        ax2.set_theta_zero_location("N")
        ax2.grid(True)
        ax2.set_title(r"$|E_{TM}|^2$ transmitted", va='bottom')
        ax2.set_ylim(0,30.5*12e3)
        ylocs = np.linspace(0,30,6)*12e3
        ax2.set_yticks(ylocs)
        ax2.set_yticklabels(ylocs/12e3)
        ax2.text(np.pi*(1.125),(30 + 7.04)*12e3,'*scale by 12e3')
        ax2.plot([np.pi/2,np.pi*3/2],\
                 np.array([1,1])*30*12e3,c='black') #surface
        
        
        
        fig.savefig('thz_6_0_5 _ %a.png'%(counter),bbox_inches = 'tight')
        ax1.cla()
        ax2.cla()
    
    counter += 1


print(time.asctime())