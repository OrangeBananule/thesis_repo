# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 22:58:22 2020

@author: Cerx
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

print('Time start:')
print(time.asctime())


def G(x,y,z):
    xcoeff = (sigma_x*np.sqrt(2*np.pi))**-1
    ycoeff = (sigma_y*np.sqrt(2*np.pi))**-1
    zcoeff = (sigma_z*np.sqrt(2*np.pi))**-1
    xe = np.e**(-0.5*(x/sigma_x)**2)
    ye = np.e**(-0.5*(y/sigma_y)**2)
    ze = np.e**(-0.5*(z/sigma_z)**2)
    return xcoeff*xe*ycoeff*ye*zcoeff*ze

def med(z):
    # return np.heaviside(z,0)*np.e**(-alpha*z) + 1-np.heaviside(z,0)
    # return np.heaviside(z,1)*np.e**(-alpha*z) + 1-np.heaviside(z,1)
    return np.heaviside(z,1)*(np.e**(-alpha*z))


np.random.seed(111)

from params3d import \
   I0, dt,v,alpha,c, zbeamwidth,sigma_x, sigma_y, sigma_z,\
                dx, dy, dzbeam, x, y, zbeam


z = zbeam
zwidth = zbeamwidth
dz = dzbeam
photon_number = np.copy(I0)

r_array = np.zeros((photon_number,4))
while photon_number > 0:
    xtemp = (x[-1] - x[0] + dx)*np.random.random() + x[0] - 0.5*dx
    ytemp = (y[-1] - y[0] + dy)*np.random.random() + y[0] - 0.5*dy
    ztemp = (z[-1] - z[0] + dz)*np.random.random() + z[0] - 0.5*dz
    Ithresh = G(xtemp,ytemp,ztemp)
    Itemp = np.random.random()*(sigma_x*np.sqrt(2*np.pi)*\
                                sigma_y*np.sqrt(2*np.pi)*\
                                sigma_z*np.sqrt(2*np.pi))**-1
    if Itemp <= Ithresh:
        photon_number -= 1
        r_array[photon_number,0] = xtemp
        r_array[photon_number,1] = xtemp
        r_array[photon_number,2] = ztemp
        r_array[photon_number,3] = Ithresh
np.save('r_array3D', r_array)

r_array = np.load('r_array3D.npy')
r_array[:,2] = r_array[:,2]-0.5*zwidth
xi = np.copy(r_array[:,0])
zi = np.copy(r_array[:,2])
temp0 = np.copy(r_array)
carrier_spawn = []


counter = 0
counter_end = 62 #(zwidth + 2/alpha)/(v*dt)
while counter < counter_end:
    t_current = counter*dt
    temp1 = temp0[temp0[:,2]  <= -c*dt]
    temp2 = temp0[temp0[:,2]  > -c*dt]
    temp2 = temp2[temp2[:,2] <= 0]
    temp3 = temp0[temp0[:,2] > 0]
    
    
    m = 0
    while m < temp3.shape[0]:
        target = temp3[m,:]
        r = np.random.random()
        if r > np.e**(-alpha*(target[2])): #absorb
            target = target.tolist()
            
            temp0 = temp0.tolist()
            temp3 = temp3.tolist()
            carrier_spawn.append([target[0],target[1],\
                                  target[2],t_current])
            temp0.remove(target)
            temp3.remove(target)
            temp0 = np.array(temp0)
            temp3 = np.array(temp3)
            if temp0.shape[0] == 0:
                break
        m += 1
    if temp0.shape[0] == 0:
        break
    
    # create image of laser photons @ t = 0
    if counter == 0:
        r_copy = np.copy(r_array)
        r_ind = np.argsort(r_copy[:,3])
        for i in range(r_ind.shape[0]):
            r = r_ind[i]
            r_copy[i,:] = r_array[r,:]
        r_array = np.copy(r_copy)
        plt.scatter(r_array[:,0],r_array[:,2],c=r_array[:,3]/np.amax(r_array[:,3]))
        plt.plot([x[0]/2,x[-1]/2],[0,0],color='black') #surface
        plt.colorbar(ticks=np.linspace(0,1,11),label='Intensity (normalized to max.)')
        plt.xticks(np.arange(-6,7,2)*1e-7,np.arange(-6,7,2)/10)
        plt.xlabel(r'x ($\mu m$)')
        plt.ylim(0 + dz,-zwidth - dz)
        plt.yticks(np.arange(-6,1,1)*0.5*1e-5,np.arange(-6,1,1)*0.5*10)
        plt.ylabel(r'z ($\mu m$)')
        plt.tight_layout
        
        filename = os.path.basename(__file__)
        # file name without extension
        filename = os.path.splitext(filename)[0]
        plt.savefig('%s_i.png'%(filename),bbox_inches='tight')
        plt.show()
        # sys.exit()
        
    # create image of laser photons @ t = counter*dt
    if counter == 25:
        r_copy = np.copy(temp0)
        r_ind = np.argsort(r_copy[:,3])
        for i in range(r_ind.shape[0]):
            r = r_ind[i]
            r_copy[i,:] = temp0[r,:]
        temp0 = np.copy(r_copy)
        #coloring exponential decay of intensity inside
        for q in range(temp0.shape[0]):
            if temp0[q,2] > 0:
                temp0[q,3] *= np.e**(-alpha*temp0[q,2])
        plt.scatter(temp0[:,0],temp0[:,2],c=temp0[:,3]/np.amax(temp0[:,3]))
        plt.colorbar(ticks=np.linspace(0,1,11),label='Intensity (normalized to max.)')
        plt.plot([x[0]/2,x[-1]/2],[0,0],color='black') #surface
        plt.xticks(np.arange(-6,7,2)*1e-7,np.arange(-6,7,2)/10)
        plt.xlabel(r'x ($\mu m$)')
        plt.ylim(0.5*zwidth/4 + 0.5*dz,-0.5*zwidth/4 - 0.5*dz)
        plt.yticks(np.arange(-3,4,1)*(1/3)*1.5e-5/4,np.arange(-3,4,1)*(1/3)*15/4)
        plt.ylabel(r'z ($\mu m$)')
        plt.tight_layout
        
        filename = os.path.basename(__file__)
        # file name without extension
        filename = os.path.splitext(filename)[0]
        plt.savefig('%s_25.png'%(filename),bbox_inches='tight')
        plt.show()
        # sys.exit()
    
    
    temp3[:,2] = temp3[:,2] + v*dt
    temp2[:,2] = v*(dt - (0-temp2[:,2])/c)
    temp1[:,2] = temp1[:,2] + c*dt
    temp3 = temp3.tolist()
    temp2 = temp2.tolist()
    temp1 = temp1.tolist()
    temp0 = np.array(temp1+temp2+temp3)
    
    counter += 1


np.save('carrier_spawn3D',np.array(carrier_spawn))

print('Time end:')
print(time.asctime())