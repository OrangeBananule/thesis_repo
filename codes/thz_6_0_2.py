# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 22:58:22 2020

@author: Cerx
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e
import time
import os

print('Time start:')
print(time.asctime())


seed = 444
np.random.seed(seed)


from params3d import \
    dt,xwidth, ywidth, zmaterialwidth, xlen, ylen, zlen, \
                zmid, dx, dy, dzmaterial, x, y, zmaterial, Z,\
                    Num_e, Num_h, me, mh, ve, vh, tR

from funcs3d import  cheby, z_avg, walk_hot_e, walk_hot_h


z = zmaterial
zwidth = zmaterialwidth
dz = dzmaterial


num_e = xlen*ylen*(zmid)*Num_e
num_h = xlen*ylen*(zmid)*Num_h
volume = xwidth*ywidth*0.5*zwidth
donor = num_e/volume


# master_array = np.load('master_array.npy')
carrier_spawn = np.load('carrier_spawn3D_%s.npy'%seed)
master_array = np.zeros((carrier_spawn.shape[0],14))
master_array[:,:4] = np.copy(carrier_spawn)
master_array[:,-1] = master_array[:,3] + tR # t_recomb = t_spawn + tR
hot_e = np.copy(master_array)
hot_e[:,6] = ve
hot_h = np.copy(master_array)
hot_h[:,6] = vh
potential = np.load('pot_f3D.npy')
pot_f = np.copy(potential)
rho_donor = np.load('rho_donor3D.npy')
dti_net = []
counter = 0
counter_end = counter + 1 + 100 #500 = 40 mins, 1500 = 2 hrs

# rho_donor = np.load('rho_donor3D.npy')
# hot_e = np.load('hot_e3D_exp.npy')
# hot_h = np.load('hot_h3D_exp.npy')
# potential = np.load('pot_f3D.npy')
# pot_f = np.copy(potential)
# rho_donor = np.load('rho_donor3D.npy')
# dti_net = list(np.load('dti_net.npy'))
# counter = int(np.load('counter_exp.npy'))
# counter_end = counter + 200 #500 = 40 mins, 1500 = 2 hrs, current = 12

fig, (ax1,ax2,ax3) = plt.subplots(1,3)
fig.set_size_inches(12,4)
fig.tight_layout(pad=2)
frames = []
rho = np.zeros((zlen,ylen,xlen))


while True:
    t_current = counter*dt
    
    # update charge density
    rho = rho*0
    rho_cold_e = np.copy(rho)
    rho_cold_h = np.copy(rho)
    rho_hot_e = np.copy(rho)
    rho_hot_h = np.copy(rho)
    
    
    
    # hot carriers
    hot_e = hot_e[hot_e[:,-1] > t_current]
    while np.any(hot_e[:,0]< x[0]-0.5*dx) == True or np.any(hot_e[:,0] > x[-1]+0.5*dx) == True:
        hot_e[:,0] = x[-1] + 0.5*dx - abs(hot_e[:,0] - (x[0] - 0.5*dx))
        hot_e[:,0] = x[0] - 0.5*dx + abs(hot_e[:,0] - (x[-1] + 0.5*dx))
    while np.any(hot_e[:,1]< y[0]-0.5*dy) == True or np.any(hot_e[:,1] > y[-1]+0.5*dy) == True:
        hot_e[:,1] = y[-1] + 0.5*dy - abs(hot_e[:,1] - (y[0] - 0.5*dy))
        hot_e[:,1] = y[0] - 0.5*dy + abs(hot_e[:,1] - (y[-1] + 0.5*dy))
    while np.any(hot_e[:,2]< z[zmid]-0.5*dz) == True or np.any(hot_e[:,2] > z[-1]+0.5*dz) == True:
        hot_e[:,2] = z[zmid] - 0.5*dz + abs(hot_e[:,2] - (z[zmid] - 0.5*dz))
        hot_e[:,2] = z[-1] + 0.5*dz - abs(hot_e[:,2] - (z[-1] + 0.5*dz))
    hot_e_eff = hot_e[hot_e[:,3] <= t_current]
    for i in range(hot_e_eff.shape[0]):
        xi = hot_e_eff[i,0]
        yi = hot_e_eff[i,1]
        zi = hot_e_eff[i,2]
        for j in x:
            if j - 0.5*dx <= xi < j + 0.5*dx:
                xi = j
        for j in y:
            if j - 0.5*dy <= yi < j + 0.5*dy:
                yi = j
        for j in z:
            if j - 0.5*dz <= zi < j + 0.5*dz:
                zi = j
        xi = x.tolist().index(xi)
        yi = y.tolist().index(yi)
        zi = z.tolist().index(zi)
        rho_hot_e[zi,yi,xi] -= 1
    
    hot_h = hot_h[hot_h[:,-1] > t_current]
    while np.any(hot_h[:,0]< x[0]-0.5*dx) == True or np.any(hot_h[:,0] > x[-1]+0.5*dx) == True:
        hot_h[:,0] = x[-1] + 0.5*dx - abs(hot_h[:,0] - (x[0] - 0.5*dx))
        hot_h[:,0] = x[0] - 0.5*dx + abs(hot_h[:,0] - (x[-1] + 0.5*dx))
    while np.any(hot_h[:,1]< y[0]-0.5*dy) == True or np.any(hot_h[:,1] > y[-1]+0.5*dy) == True:
        hot_h[:,1] = y[-1] + 0.5*dy - abs(hot_h[:,1] - (y[0] - 0.5*dy))
        hot_h[:,1] = y[0] - 0.5*dy + abs(hot_h[:,1] - (y[-1] + 0.5*dy))
    while np.any(hot_h[:,2]< z[zmid]-0.5*dz) == True or np.any(hot_h[:,2] > z[-1]+0.5*dz) == True:
        hot_h[:,2] = z[zmid] - 0.5*dz + abs(hot_h[:,2] - (z[zmid] - 0.5*dz))
        hot_h[:,2] = z[-1] + 0.5*dz - abs(hot_h[:,2] - (z[-1] + 0.5*dz))
    hot_h_eff = hot_h[hot_h[:,3] <= t_current]
    for i in range(hot_h_eff.shape[0]):
        xi = hot_h_eff[i,0]
        yi = hot_h_eff[i,1]
        zi = hot_h_eff[i,2]
        for j in x:
            if j - 0.5*dx <= xi < j + 0.5*dx:
                xi = j
        for j in y: 
            if j - 0.5*dy <= yi < j + 0.5*dy:
                yi = j
        for j in z:
            if j - 0.5*dz <= zi < j + 0.5*dz:
                zi = j
        xi = x.tolist().index(xi)
        yi = y.tolist().index(yi)
        zi = z.tolist().index(zi)
        rho_hot_h[zi,yi,xi] += 1
    rho_hot_e = rho_hot_e*e*(dx*dy*dz)**-1
    rho_hot_h = rho_hot_h*e*(dx*dy*dz)**-1
    rho = (rho_hot_e + rho_hot_h)
    
    
    
    if counter%(10) == 0:
        # plot electrons
        x_hot_e = hot_e_eff[:,0]
        y_hot_e = hot_e_eff[:,1]
        z_hot_e = hot_e_eff[:,2]
        frame2hote = ax1.scatter(list(x_hot_e),\
                              list(z_hot_e), linewidth=0.5, marker='.', color='lime',edgecolor='black',label='pht_e')
        ax1.plot([x[0]-0.5*dx,x[-1]+0.5*dx],[0,0], color='black',label='surface')
        ax1.legend(loc='upper right')
        
        #format
        ax1.set_xlim(x[0],x[-1])
        xlocs = np.arange(-2,3,1)*xwidth/4
        ax1.set_xticks(xlocs)
        ax1.set_xticklabels(np.round(xlocs*1e6,2))
        ax1.set_xlabel('x ($\mu$m)')
        ax1.set_ylim(Z[-1],Z[0])
        zlocs = np.arange(-4,5,1)*0.5*zwidth/4
        ax1.set_yticks(zlocs)
        ax1.set_yticklabels(np.round(zlocs*1e6,2))
        ax1.set_ylabel('z ($\mu$m)')
        
        # plot electrons
        x_hot_h = hot_h_eff[:,0]
        y_hot_h = hot_h_eff[:,1]
        z_hot_h = hot_h_eff[:,2]
        frame2h = ax2.scatter(list(x_hot_h),\
                              list(z_hot_h), linewidth=0.5, marker='.', facecolor='none',edgecolor='r',label='pht_h')
        ax2.plot([x[0]-0.5*dx,x[-1]+0.5*dx],[0,0], color='black',label='surface')
        ax2.legend(loc='upper right')
        
        #format
        ax2.set_xlim(x[0],x[-1])
        xlocs = np.arange(-2,3,1)*xwidth/4
        ax2.set_xticks(xlocs)
        ax2.set_xticklabels(np.round(xlocs*1e6,2))
        ax2.set_xlabel('x ($\mu$m)')
        ax2.set_ylim(Z[-1],Z[0])
        zlocs = np.arange(-4,5,1)*0.5*zwidth/4
        ax2.set_yticks(zlocs)
        ax2.set_yticklabels(np.round(zlocs*1e6,2))
        ax2.set_ylabel('z ($\mu$m)')
        
        # plot net potential along z
        zpot = z_avg(pot_f)[zmid:]
        zpot = (zpot)/(np.amax(zpot))
        frame3, = ax3.plot(zpot,z[zmid:],color='blue',label='electric potential')
        ztotale = -z_avg(-rho_donor + rho_hot_e)[zmid:]
        ztotale = (ztotale/abs(np.amax(ztotale)))/2
        frame4, = ax3.plot(ztotale,z[zmid:],color='lime',label='n_electrons')
        ztotalh = z_avg(rho_hot_h + rho_donor)[zmid:]
        ztotalh = (ztotalh/abs(np.amax(ztotalh)))/2
        frame5, = ax3.plot(ztotalh,z[zmid:],color = 'r'\
                           ,label='n_holes + n_donors',linestyle='--')
            
        # format
        ax3.set_ylim(z[0]-0.5*dz,z[-1]+0.5*dz)
        ylocs = np.linspace(z[0]-0.5*dz,z[-1]+0.5*dz,9)
        ax3.set_yticks(ylocs)
        ax3.set_yticklabels(np.round(ylocs*1e6,2))
        ax3.set_ylabel('z ($\mu$m)')
        ax3.invert_yaxis()
        ax3.set_xlim(0,1.2)
        ax3.plot([0,1.2],[0,0],color='black')
        ax3.legend()
        
        filename = os.path.basename(__file__)
        # file name without extension
        filename = os.path.splitext(filename)[0]
        fig.savefig('%s _ %a_%c.png'%(filename,counter,seed),bbox_inches = 'tight')
        ax1.cla()
        ax2.cla()
        ax3.cla()
    
    print(counter)
    if counter == counter_end:
        break
    
    #record dti_net
    dtf_temp = np.sum(hot_e_eff[:,7:10],axis=0)*e**2/me + np.sum(hot_h_eff[:,7:10],axis=0)*e**2/mh
    dti_net.append(dtf_temp)
    
    # solve potential
    pot_i = np.copy(pot_f)
    pot_f = cheby(potential,rho,dx,dy,dz)
    
    #move charges
    hot_e_eff = walk_hot_e(hot_e_eff,pot_f,t_current,dt)
    hot_h_eff = walk_hot_h(hot_h_eff,pot_f,t_current,dt)
    hot_e[hot_e[:,3] <= t_current] = hot_e_eff
    hot_h[hot_h[:,3] <= t_current] = hot_h_eff
    
    
    counter += 1


dti_net = np.array(dti_net)
np.save('dti_net_%s'%seed,dti_net)

np.save('pot_f3D_exp',pot_f)
np.save('hot_e3D_exp',hot_e)
np.save('hot_h3D_exp',hot_h)
np.save('counter_exp',counter)

print(time.asctime())