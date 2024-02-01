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

seeds = [111,222,333,444,555,666,777,888,999,101010]

for g in seeds:
    np.random.seed(g)
    
    # create potential
    from params3d import \
            xwidth, ywidth, zmaterialwidth, xlen, ylen, zlen, \
                    ymid, zmid, dx, dy, dzmaterial, zmaterial, Num_e, Num_h
    from funcs3d import  cheby


    z = zmaterial
    zwidth = zmaterialwidth
    dz = dzmaterial

    num_e = xlen*ylen*(zmid)*Num_e
    num_h = xlen*ylen*(zmid)*Num_h
    volume = xwidth*ywidth*0.5*zwidth
    donor = num_e/volume #lower = potential more diffuse


    potential = np.zeros((zlen,ylen,xlen))
    rho = np.zeros((zlen,ylen,xlen))
    rho_donor = np.zeros((zlen,ylen,xlen))



    fig, (ax1,ax2,ax3) = plt.subplots(1,3)
    fig.set_size_inches(16,4)
    fig.tight_layout(pad=2)
    potential[zmid,:,:] = 1
    frame1 = ax1.imshow(potential[:,ymid,:], cmap ='jet')
    cb = fig.colorbar(frame1,ax=ax1)

    # solve potential
    pot_f = cheby(potential,rho,dx,dy,dz)

    print('Time end:')
    print(time.asctime())

    np.save('pot_f3D',pot_f)
    np.save('rho_donor3D',rho_donor)
    
    
    
    #create photons
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
            r_array[photon_number,3] = Itemp
    np.save('r_array3D_%s'%g, r_array)
    
    r_array = np.load('r_array3D_%s.npy'%g)
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
            
            
            plt.savefig('thz_6_0_1_i_%s.png'%(g),bbox_inches='tight')
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
            
            plt.savefig('thz_6_0_1_%a_%s.png'%(counter,g),bbox_inches='tight')
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
    
    
    np.save('carrier_spawn3D_%s'%g,np.array(carrier_spawn))
    
    print('Time end:')
    print(time.asctime())
    
    from scipy.constants import e
    
    print('Time start:')
    print(time.asctime())
    
    
    
    #charges in GaAs create di/dt
    
    from params3d import \
        dt,xwidth, ywidth, zmaterialwidth, xlen, ylen, zlen, tR,\
                    zmid, dx, dy, dzmaterial, x, y, zmaterial, Z,\
                        Num_e, Num_h, me, mh, ve,vh
    
    from funcs3d import  cheby, z_avg, walk_hot_e, walk_hot_h
    
    
    z = zmaterial
    zwidth = zmaterialwidth
    dz = dzmaterial
    
    
    num_e = xlen*ylen*(zmid)*Num_e
    num_h = xlen*ylen*(zmid)*Num_h
    volume = xwidth*ywidth*0.5*zwidth
    donor = num_e/volume
    
    
    # master_array = np.load('master_array.npy')
    carrier_spawn = np.load('carrier_spawn3D_%s.npy'%g)
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
    counter_end = counter + 1 + 3*500 #500 = 40 mins, 1500 = 2 hrs
    
    # rho_donor = np.load('rho_donor3D_%s.npy'%g)
    # hot_e = np.load('hot_e3D_exp_%s.npy'%g)
    # hot_h = np.load('hot_h3D_exp_%s.npy'%g)
    # potential = np.load('pot_f3D_exp_%s.npy'%g)
    # pot_f = np.copy(potential)
    # rho_donor = np.load('rho_donor3D.npy')
    # dti_net = list(np.load('dti_net_%s.npy'%g))
    # counter = int(np.load('counter_exp.npy'))
    # counter_end = counter + 2*500 #500 = 40 mins, 1500 = 2 hrs, current = 12
    
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
        while np.any(hot_e[:,0]< x[0]-0.5*dx) == True \
            or np.any(hot_e[:,0] > x[-1]+0.5*dx) == True:
            hot_e[:,0] = x[-1] + 0.5*dx - abs(hot_e[:,0] - (x[0] - 0.5*dx))
            hot_e[:,0] = x[0] - 0.5*dx + abs(hot_e[:,0] - (x[-1] + 0.5*dx))
        while np.any(hot_e[:,1]< y[0]-0.5*dy) == True \
            or np.any(hot_e[:,1] > y[-1]+0.5*dy) == True:
            hot_e[:,1] = y[-1] + 0.5*dy - abs(hot_e[:,1] - (y[0] - 0.5*dy))
            hot_e[:,1] = y[0] - 0.5*dy + abs(hot_e[:,1] - (y[-1] + 0.5*dy))
        while np.any(hot_e[:,2]< z[zmid]-0.5*dz) == True \
            or np.any(hot_e[:,2] > z[-1]+0.5*dz) == True:
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
        while np.any(hot_h[:,0]< x[0]-0.5*dx) == True \
            or np.any(hot_h[:,0] > x[-1]+0.5*dx) == True:
            hot_h[:,0] = x[-1] + 0.5*dx - abs(hot_h[:,0] - (x[0] - 0.5*dx))
            hot_h[:,0] = x[0] - 0.5*dx + abs(hot_h[:,0] - (x[-1] + 0.5*dx))
        while np.any(hot_h[:,1]< y[0]-0.5*dy) == True \
            or np.any(hot_h[:,1] > y[-1]+0.5*dy) == True:
            hot_h[:,1] = y[-1] + 0.5*dy - abs(hot_h[:,1] - (y[0] - 0.5*dy))
            hot_h[:,1] = y[0] - 0.5*dy + abs(hot_h[:,1] - (y[-1] + 0.5*dy))
        while np.any(hot_h[:,2]< z[zmid]-0.5*dz) == True \
            or np.any(hot_h[:,2] > z[-1]+0.5*dz) == True:
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
            frame2hote = ax1.scatter(list(x_hot_e),list(z_hot_e),\
                                     linewidth=0.5, marker='.', color='lime',\
                                         edgecolor='black',label='pht_e')
            ax1.plot([x[0]-0.5*dx,x[-1]+0.5*dx],[0,0],\
                     color='black',label='surface')
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
            frame2h = ax2.scatter(list(x_hot_h),list(z_hot_h),\
                                  linewidth=0.5, marker='.', facecolor='none',\
                                      edgecolor='r',label='pht_h')
            ax2.plot([x[0]-0.5*dx,x[-1]+0.5*dx],[0,0],\
                     color='black',label='surface')
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
            frame3, = ax3.plot(zpot,z[zmid:],color='blue',\
                               label='electric potential')
            ztotale = -z_avg(-rho_donor + rho_hot_e)[zmid:]
            ztotale = (ztotale/abs(np.amax(ztotale)))/2
            frame4, = ax3.plot(ztotale,z[zmid:],color='lime',\
                               label='n_electrons')
            ztotalh = z_avg(rho_hot_h + rho_donor)[zmid:]
            ztotalh = (ztotalh/abs(np.amax(ztotalh)))/2
            frame5, = ax3.plot(ztotalh,z[zmid:],color = 'r',\
                               label='n_holes + n_donors',linestyle='--')
                
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
            
            fig.savefig('thz_6_0_2 _ %a_%s.png'%(counter,g),bbox_inches='tight')
            ax1.cla()
            ax2.cla()
            ax3.cla()
        
        print(counter)
        if counter == counter_end:
            break
        
        #record dti_net
        dtf_temp = np.sum(hot_e_eff[:,7:10],axis=0)*e**2/me +\
            np.sum(hot_h_eff[:,7:10],axis=0)*e**2/mh
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
    np.save('dti_net_%s'%g,dti_net)
    
    np.save('pot_f3D_exp_%s'%g,pot_f)
    np.save('hot_e3D_exp_%s'%g,hot_e)
    np.save('hot_h3D_exp_%s'%g,hot_h)
    np.save('counter_exp_%s'%g,counter)
    
    print(time.asctime())
    
    
    
    from params3d import n
    from funcs3d import  E_thz
    
    
    print('Time start:')
    print(time.asctime())
    
    
    
    fig, (ax1,ax2) = plt.subplots(1,2,subplot_kw={'projection': 'polar'})
    fig.set_size_inches(10,4)
    fig.tight_layout(pad=2)
    
    
    dti_net = np.load('dti_net_%s.npy'%g)
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
            frame1, = ax1.plot(theta_e, prefac*E_TE**2,color='blue')
            frame2, = ax2.plot(theta_e, prefac*E_TM**2,color='blue')
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
            
            
            
            fig.savefig('thz_6_0_3 _ %a_%s.png'%(counter,g),bbox_inches = 'tight')
            ax1.cla()
            ax2.cla()
            
        elif counter%(10) == 0 and counter > 800:
            #plot E_TE, E_TM
            E_TE,E_TM,E_TMsin,E_TMcos = E_thz(dti_temp,theta_i,theta_e)
            prefac = np.cos(theta_e)/(n*np.cos(theta_i))
            frame1, = ax1.plot(theta_e, prefac*E_TE**2,color='blue')
            frame2, = ax2.plot(theta_e, prefac*E_TM**2,color='blue')
            
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
            
            
            fig.savefig('thz_6_0_3 _ %a_%s.png'%(counter,g),bbox_inches='tight')
            ax1.cla()
            ax2.cla()
        
        counter += 1
    
    
    print(time.asctime())
    
    
    
    #create power plot
    from numpy.fft import rfft, irfft, rfftfreq
    
    from params3d import \
        n, epsilon, dt,\
            tR, xwidth, ywidth, zmaterialwidth, \
                xlen, ylen, zlen, \
                    zmid, dx, dy, dzmaterial, x, y, zmaterial, \
                        Num_e, Num_h, phi_s, kbT,\
                            me, mh, ve, vh, ve_therm
    
    
    dti_net = np.load('dti_net_%s.npy'%g)
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
    
    
    fig.savefig('thz_6_0_4_%s.png'%(g),bbox_inches='tight')
    
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
    
    fig.savefig('thz_6_0_4_smooth_%s.png'%(g),bbox_inches='tight')
    
    fig,ax3 = plt.subplots(1,1)
    fig.set_size_inches(4,4)
    fig.tight_layout(pad=2)
    
    print(time.asctime())
    
    
    print('Time end:')
    print(time.asctime())

#os.system(f'shutdown /s /t 10')