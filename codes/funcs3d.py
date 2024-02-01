# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 22:58:22 2020

@author: Cerx
"""

import numpy as np
from scipy.constants import e, epsilon_0, m_e, Boltzmann
import sys
import os

from params3d import \
    phi_s, zmid, donor, epsilon, kbT, \
        dx, dy, dzmaterial, \
            x, y, zmaterial, me, mh, alpha


z = zmaterial
dz = dzmaterial      

def cheby(pot,rho,dx,dy,dz): #solves del_sq(pot) = -rho
    sizex = pot.shape[2]
    sizey = pot.shape[1]
    sizez = pot.shape[0]
    potential = np.copy(pot)
    
    z_bulk = 1/alpha
    for k in range(zmid,z.shape[0]):
        if z[k]-0.5*dz <= z_bulk < z[k]+0.5*dz:
            z_bulk_index = k
    # bound con
    potential[zmid,:,:] = phi_s
    # pot[zmid,:,:] = phi_s
    w = 1
    spec_r = 0.33333*(np.cos(np.pi/potential.shape[0])\
                  +np.cos(np.pi/potential.shape[1])\
                  +np.cos(np.pi/potential.shape[2]))
        
    
    index_set = []
    for k in range(potential.shape[0]):
        for i in range(potential.shape[2]):
            for j in range(potential.shape[1]):
                if k%2 == 0:
                    if (i%2 == 0 and j%2 == 0) or (i%2 == 1 and j%2 == 1):
                        index_set.append((i,j,k))
                elif k%2 == 1:
                    if (i%2 == 1 and j%2 == 0) or (i%2 == 0 and j%2 == 1):
                        index_set.append((i,j,k))
    for k in range(potential.shape[0]):
        for i in range(potential.shape[2]):
            for j in range(potential.shape[1]):
                if k%2 == 1:
                    if (i%2 == 0 and j%2 == 0) or (i%2 == 1 and j%2 == 1):
                        index_set.append((i,j,k))
                elif k%2 == 0:
                    if (i%2 == 1 and j%2 == 0) or (i%2 == 0 and j%2 == 1):
                        index_set.append((i,j,k))
    
    
    while True:
        pot_i = np.copy(potential)
        for m in index_set:
            i,j,k = m[0],m[1],m[2]
            
            if k == zmid:
                continue
            
            elif k > z_bulk_index:
                potential[k,j,i] = potential[k-1,j,i]
            
            elif zmid < k <= z_bulk_index:
                if i == 0:
                    if j == 0:
                        plx = potential[k,0,-1]
                        prx = potential[k,0,+1]
                        ply = potential[k,-1,0]
                        pry = potential[k,+1,0]
                        pd = potential[k+1,0,0]
                        pu = potential[k-1,0,0]
                    elif j == sizey-1:
                        plx = potential[k,-1,-1]
                        prx = potential[k,-1,+1]
                        ply = potential[k,-2,0]
                        pry = potential[k,0,0]
                        pd = potential[k+1,-1,0]
                        pu = potential[k-1,-1,0]
                    elif 0 < j < sizey-1:
                        plx = potential[k,j,-1]
                        prx = potential[k,j,+1]
                        ply = potential[k,j-1,0]
                        pry = potential[k,j+1,0]
                        pd = potential[k+1,j,0]
                        pu = potential[k-1,j,0]
                elif i == sizex-1:
                    if j == 0:
                        plx = potential[k,0,-2]
                        prx = potential[k,0,0]
                        ply = potential[k,-1,-1]
                        pry = potential[k,+1,-1]
                        pd = potential[k+1,0,-1]
                        pu = potential[k-1,0,-1]
                    elif j == sizey-1:
                        plx = potential[k,-1,-2]
                        prx = potential[k,-1,0]
                        ply = potential[k,-2,-1]
                        pry = potential[k,0,-1]
                        pd = potential[k+1,-1,-1]
                        pu = potential[k-1,-1,-1]
                    elif 0 < j < sizey-1:
                        plx = potential[k,j,-2]
                        prx = potential[k,j,0]
                        ply = potential[k,j-1,-1]
                        pry = potential[k,j+1,-1]
                        pd = potential[k+1,j,-1]
                        pu = potential[k-1,j,-1]
                elif 0 < i < sizex-1:
                    if j == 0:
                        plx = potential[k,0,i-1]
                        prx = potential[k,0,i+1]
                        ply = potential[k,-1,i]
                        pry = potential[k,+1,i]
                        pd = potential[k+1,0,i]
                        pu = potential[k-1,0,i]
                    elif j == sizey-1:
                        plx = potential[k,-1,i-1]
                        prx = potential[k,-1,i+1]
                        ply = potential[k,-2,i]
                        pry = potential[k,0,i]
                        pd = potential[k+1,-1,i]
                        pu = potential[k-1,-1,i]
                    elif 0 < j < sizey-1:
                        plx = potential[k,j,i-1]
                        prx = potential[k,j,i+1]
                        ply = potential[k,j-1,i]
                        pry = potential[k,j+1,i]
                        pd = potential[k+1,j,i]
                        pu = potential[k-1,j,i]
                pc = potential[k,j,i]
                coeff = (2/(dx**2) + 2/(dy**2) + 2/(dz**2))**-1
                # solve -delsq(V) = rho/epsilon
                potential[k,j,i] = w*coeff*((plx + prx)/(dx**2)\
                                          + (ply + pry)/(dy**2)\
                                          + (pd + pu)/(dz**2)\
                                          + e*donor/epsilon\
                                            + (rho[k,j,i])/epsilon)\
                                              - (w-1)*pc
                                  
                
        pot_f = np.copy(potential)
        
        w = (1-0.25*w*spec_r**2)**-1
        
        if np.amax(abs(pot_f-pot_i))<1e-6:
            break
        
    return pot_f

def grad(pot):
    xsize = pot.shape[2]
    ysize = pot.shape[1]
    zsize = pot.shape[0]
    out = np.zeros((zsize, ysize, xsize,3))
    
    for i in range(0,xsize):
        for j in range(0,ysize):
            for k in range(zmid,zsize):
                if zmid < k < zsize-1:
                    if 0 < i < xsize-1:
                        if 0 < j < ysize-1:
                            out[k,j,i,0] = (pot[k,j,i+1] - pot[k,j,i-1])*(2*dx)**-1
                            out[k,j,i,1] = (pot[k,j+1,i] - pot[k,j-1,i])*(2*dy)**-1
                            out[k,j,i,2] = (pot[k+1,j,i] - pot[k-1,j,i])*(2*dz)**-1
                        elif j == 0:
                            out[k,j,i,0] = (pot[k,0,i+1] - pot[k,0,i-1])*(2*dx)**-1
                            out[k,j,i,1] = (pot[k,1,i] - pot[k,-1,i])*(2*dy)**-1
                            out[k,j,i,2] = (pot[k+1,0,i] - pot[k-1,0,i])*(2*dz)**-1
                        elif j == ysize-1:
                            out[k,j,i,0] = (pot[k,-1,i+1] - pot[k,-1,i-1])*(2*dx)**-1
                            out[k,j,i,1] = (pot[k,0,i] - pot[k,-2,i])*(2*dy)**-1
                            out[k,j,i,2] = (pot[k+1,-1,i] - pot[k-1,-1,i])*(2*dz)**-1
                    elif i == 0:
                        if 0 < j < ysize-1:
                            out[k,j,i,0] = (pot[k,j,1] - pot[k,j,-1])*(2*dx)**-1
                            out[k,j,i,1] = (pot[k,j+1,0] - pot[k,j-1,0])*(2*dy)**-1
                            out[k,j,i,2] = (pot[k+1,j,0] - pot[k-1,j,0])*(2*dz)**-1
                        elif j == 0:
                            out[k,j,i,0] = (pot[k,0,1] - pot[k,0,-1])*(2*dx)**-1
                            out[k,j,i,1] = (pot[k,1,0] - pot[k,-1,0])*(2*dy)**-1
                            out[k,j,i,2] = (pot[k+1,0,0] - pot[k-1,0,0])*(2*dz)**-1
                        elif j == ysize-1:
                            out[k,j,i,0] = (pot[k,-1,1] - pot[k,-1,-1])*(2*dx)**-1
                            out[k,j,i,1] = (pot[k,0,0] - pot[k,-2,0])*(2*dy)**-1
                            out[k,j,i,2] = (pot[k+1,-1,0] - pot[k-1,-1,0])*(2*dz)**-1
                    elif i == xsize-1:
                        if 0 < j < ysize-1:
                            out[k,j,i,0] = (pot[k,j,0] - pot[k,j,-2])*(2*dx)**-1
                            out[k,j,i,1] = (pot[k,j+1,-1] - pot[k,j-1,-1])*(2*dy)**-1
                            out[k,j,i,2] = (pot[k+1,j,-1] - pot[k-1,j,-1])*(2*dz)**-1
                        elif j == 0:
                            out[k,j,i,0] = (pot[k,0,0] - pot[k,0,-2])*(2*dx)**-1
                            out[k,j,i,1] = (pot[k,1,-1] - pot[k,-1,-1])*(2*dy)**-1
                            out[k,j,i,2] = (pot[k+1,0,-1] - pot[k-1,0,-1])*(2*dz)**-1
                        elif j == ysize-1:
                            out[k,j,i,0] = (pot[k,-1,0] - pot[k,-1,-2])*(2*dx)**-1
                            out[k,j,i,1] = (pot[k,0,-1] - pot[k,-2,-1])*(2*dy)**-1
                            out[k,j,i,2] = (pot[k+1,-1,-1] - pot[k-1,-1,-1])*(2*dz)**-1
                elif k == zmid:
                    if 0 < i < xsize-1:
                        if 0 < j < ysize-1:
                            out[k,j,i,0] = (pot[k,j,i+1] - pot[k,j,i-1])*(2*dx)**-1
                            out[k,j,i,1] = (pot[k,j+1,i] - pot[k,j-1,i])*(2*dy)**-1
                            out[k,j,i,2] = (pot[k+1,j,i] - pot[zmid,j,i])*(dz)**-1
                        elif j == 0:
                            out[k,j,i,0] = (pot[k,0,i+1] - pot[k,0,i-1])*(2*dx)**-1
                            out[k,j,i,1] = (pot[k,1,i] - pot[k,-1,i])*(2*dy)**-1
                            out[k,j,i,2] = (pot[k+1,j,i] - pot[zmid,j,i])*(dz)**-1
                        elif j == ysize-1:
                            out[k,j,i,0] = (pot[k,-1,i+1] - pot[k,-1,i-1])*(2*dx)**-1
                            out[k,j,i,1] = (pot[k,0,i] - pot[k,-2,i])*(2*dy)**-1
                            out[k,j,i,2] = (pot[k+1,j,i] - pot[zmid,j,i])*(dz)**-1
                    elif i == 0:
                        if 0 < j < ysize-1:
                            out[k,j,i,0] = (pot[k,j,1] - pot[k,j,-1])*(2*dx)**-1
                            out[k,j,i,1] = (pot[k,j+1,0] - pot[k,j-1,0])*(2*dy)**-1
                            out[k,j,i,2] = (pot[k+1,j,0] - pot[zmid,j,0])*(dz)**-1
                        elif j == 0:
                            out[k,j,i,0] = (pot[k,0,1] - pot[k,0,-1])*(2*dx)**-1
                            out[k,j,i,1] = (pot[k,1,0] - pot[k,-1,0])*(2*dy)**-1
                            out[k,j,i,2] = (pot[k+1,0,0] - pot[zmid,0,0])*(dz)**-1
                        elif j == ysize-1:
                            out[k,j,i,0] = (pot[k,-1,1] - pot[k,-1,-1])*(2*dx)**-1
                            out[k,j,i,1] = (pot[k,0,0] - pot[k,-2,0])*(2*dy)**-1
                            out[k,j,i,2] = (pot[k+1,-1,0] - pot[zmid,-1,0])*(dz)**-1
                    elif i == xsize-1:
                        if 0 < j < ysize-1:
                            out[k,j,i,0] = (pot[k,j,0] - pot[k,j,-2])*(2*dx)**-1
                            out[k,j,i,1] = (pot[k,j+1,-1] - pot[k,j-1,-1])*(2*dy)**-1
                            out[k,j,i,2] = (pot[k+1,j,-1] - pot[zmid,j,-1])*(dz)**-1
                        elif j == 0:
                            out[k,j,i,0] = (pot[k,0,0] - pot[k,0,-2])*(2*dx)**-1
                            out[k,j,i,1] = (pot[k,1,-1] - pot[k,-1,-1])*(2*dy)**-1
                            out[k,j,i,2] = (pot[k+1,0,-1] - pot[zmid,0,-1])*(dz)**-1
                        elif j == ysize-1:
                            out[k,j,i,0] = (pot[k,-1,0] - pot[k,-1,-2])*(2*dx)**-1
                            out[k,j,i,1] = (pot[k,0,-1] - pot[k,-2,-1])*(2*dy)**-1
                            out[k,j,i,2] = (pot[k+1,-1,-1] - pot[zmid,-1,-1])*(dz)**-1
                elif k == zsize-1:
                    if 0 < i < xsize-1:
                        if 0 < j < ysize-1:
                            out[k,j,i,0] = (pot[-1,j,i+1] - pot[k,j,i-1])*(2*dx)**-1
                            out[k,j,i,1] = (pot[-1,j+1,i] - pot[k,j-1,i])*(2*dy)**-1
                            out[k,j,i,2] = (pot[-1,j,i] - pot[-2,j,i])*(dz)**-1
                        elif j == 0:
                            out[k,j,i,0] = (pot[-1,0,i+1] - pot[-1,0,i-1])*(2*dx)**-1
                            out[k,j,i,1] = (pot[-1,1,i] - pot[-1,-1,i])*(2*dy)**-1
                            out[k,j,i,2] = (pot[-1,0,i] - pot[-1,0,i])*(dz)**-1
                        elif j == ysize-1:
                            out[k,j,i,0] = (pot[-1,-1,i+1] - pot[-1,-1,i-1])*(2*dx)**-1
                            out[k,j,i,1] = (pot[-1,0,i] - pot[-1,-2,i])*(2*dy)**-1
                            out[k,j,i,2] = (pot[-1,-1,i] - pot[-2,-1,i])*(dz)**-1
                    elif i == 0:
                        if 0 < j < ysize-1:
                            out[k,j,i,0] = (pot[-1,j,1] - pot[-1,j,-1])*(2*dx)**-1
                            out[k,j,i,1] = (pot[-1,j+1,0] - pot[-1,j-1,0])*(2*dy)**-1
                            out[k,j,i,2] = (pot[-1,j,0] - pot[-2,j,0])*(dz)**-1
                        elif j == 0:
                            out[k,j,i,0] = (pot[-1,0,1] - pot[-1,0,-1])*(2*dx)**-1
                            out[k,j,i,1] = (pot[-1,1,0] - pot[-1,-1,0])*(2*dy)**-1
                            out[k,j,i,2] = (pot[-1,0,0] - pot[-2,0,0])*(dz)**-1
                        elif j == ysize-1:
                            out[k,j,i,0] = (pot[-1,-1,1] - pot[-1,-1,-1])*(2*dx)**-1
                            out[k,j,i,1] = (pot[-1,0,0] - pot[-1,-2,0])*(2*dy)**-1
                            out[k,j,i,2] = (pot[-1,-1,0] - pot[-2,-1,0])*(dz)**-1
                    elif i == xsize-1:
                        if 0 < j < ysize-1:
                            out[k,j,i,0] = (pot[-1,j,0] - pot[-1,j,-2])*(2*dx)**-1
                            out[k,j,i,1] = (pot[-1,j+1,-1] - pot[-1,j-1,-1])*(2*dy)**-1
                            out[k,j,i,2] = (pot[-1,j,-1] - pot[-2,j,-1])*(2*dz)**-1
                        elif j == 0:
                            out[k,j,i,0] = (pot[-1,0,0] - pot[-1,0,-2])*(2*dx)**-1
                            out[k,j,i,1] = (pot[-1,1,-1] - pot[-1,-1,-1])*(2*dy)**-1
                            out[k,j,i,2] = (pot[-1,0,-1] - pot[-2,0,-1])*(2*dz)**-1
                        elif j == ysize-1:
                            out[k,j,i,0] = (pot[-1,-1,0] - pot[-1,-1,-2])*(2*dx)**-1
                            out[k,j,i,1] = (pot[-1,0,-1] - pot[-1,-2,-1])*(2*dy)**-1
                            out[k,j,i,2] = (pot[-1,-1,-1] - pot[-2,-1,-1])*(2*dz)**-1
    
    # out[:,:,2] = -out[:,:,2]
    return out


def scatter(vel,P):
    p = np.random.random()
    if p < P:
        vel_norm = np.linalg.norm(vel)
        phi = np.random.random()*2*np.pi
        theta = np.random.random()*np.pi
        xvel = np.sin(theta)*np.cos(phi)
        yvel = np.sin(theta)*np.sin(phi)
        zvel = np.cos(theta)
        vel = vel_norm*np.array([xvel,yvel,zvel])
        #vel = 0 #for inelastic scattering
    return vel


def walk_hot_e(r,pot_f,t_current,dt):
    E_field = -grad(pot_f)
    for l in range(r.shape[0]):
        i = r[l,0]
        j = r[l,1]
        k = r[l,2]
        for m in x:
            if m - 0.5*dx <= i < m + 0.5*dx:
                i = m
        for m in y:
            if m - 0.5*dy <= j < m + 0.5*dy:
                j = m
        for m in z[zmid:]:
            if m - 0.5*dz <= k < m + 0.5*dz:
                k = m
        i = x.tolist().index(i)
        j = y.tolist().index(j)
        k = z.tolist().index(k)
        r[l,7:10] = E_field[k,j,i,:]

    #r[:,4:7] -> v, r[:,7:10] -> E
    for i in range(r.shape[0]):
        if r[i,3] <= t_current:
            if t_current > 0:
                #eqn of motion
                r[i,:3] = r[i,:3] + r[i,4:7]*dt
                r[i,4:7] = r[i,4:7] - ((e/me)*r[i,7:10])*dt
                
                # scatter
                r[i,4:7] = scatter(r[i,4:7],0.0005)
    
    # bc
    for i in range(r.shape[0]):
        while r[i,0] <= x[0]-0.5*dx or r[i,0] > x[-1]+0.5*dx:
            if r[i,0] <= x[0]-0.5*dx:
                r[i,0] = x[-1]+0.5*dx - abs(r[i,0]-(x[0]-0.5*dx))
            elif r[i,0] > x[-1]+0.5*dx:
                r[i,0] = x[0]-0.5*dx + abs(r[i,0]-(x[-1]+0.5*dx))
        while r[i,1] <= y[0]-0.5*dy or r[i,1] > y[-1]+0.5*dy:
            if r[i,1] <= y[0]-0.5*dy:
                r[i,1] = y[-1]+0.5*dy - abs(r[i,1]-(y[0]-0.5*dy))
            elif r[i,1] > y[-1]+0.5*dy:
                r[i,1] = y[0]-0.5*dy + abs(r[i,1]-(y[-1]+0.5*dy))
        while r[i,2] < z[zmid] - 0.5*dz:
            r[i,2] = z[zmid] - 0.5*dz + abs(r[i,2]-(z[zmid]-0.5*dz))
            r[i,6] = -r[i,6]
        while r[i,2] >= z[-1] + 0.5*dz:
            r[i,2] = z[-1]+ 0.5*dz - abs(r[i,2]-(z[-1]+0.5*dz))
            r[i,6] = -r[i,6]
    return r

def walk_hot_h(r,pot_f,t_current,dt):
    E_field = -grad(pot_f)
    for l in range(r.shape[0]):
        i = r[l,0]
        j = r[l,1]
        k = r[l,2]
        for m in x:
            if m - 0.5*dx <= i < m + 0.5*dx:
                i = m
        for m in y:
            if m - 0.5*dy <= j < m + 0.5*dy:
                j = m
        for m in z[zmid:]:
            if m - 0.5*dz <= k < m + 0.5*dz:
                k = m
        i = x.tolist().index(i)
        j = y.tolist().index(j)
        k = z.tolist().index(k)
        r[l,7:10] = E_field[k,j,i,:]
        
    #r[:,4:7] -> v, r[:,7:10] -> E
    for i in range(r.shape[0]):
        if r[i,3] <= t_current:
            if t_current > 0:
                #eqn of motion
                r[i,:3] = r[i,:3] + r[i,4:7]*dt
                r[i,4:7] = r[i,4:7] + ((e/mh)*r[i,7:10])*dt
                
                # scatter
                r[i,4:7] = scatter(r[i,4:7],0.0005)
    
    # bc
    for i in range(r.shape[0]):
        while r[i,0] <= x[0]-0.5*dx or r[i,0] > x[-1]+0.5*dx:
            if r[i,0] <= x[0]-0.5*dx:
                r[i,0] = x[-1]+0.5*dx - abs(r[i,0]-(x[0]-0.5*dx))
            elif r[i,0] > x[-1]+0.5*dx:
                r[i,0] = x[0]-0.5*dx + abs(r[i,0]-(x[-1]+0.5*dx))
        while r[i,1] <= y[0]-0.5*dy or r[i,1] > y[-1]+0.5*dy:
            if r[i,1] <= y[0]-0.5*dy:
                r[i,1] = y[-1]+0.5*dy - abs(r[i,1]-(y[0]-0.5*dy))
            elif r[i,1] > y[-1]+0.5*dy:
                r[i,1] = y[0]-0.5*dy + abs(r[i,1]-(y[-1]+0.5*dy))
        while r[i,2] < z[zmid] - 0.5*dz:
            r[i,2] = z[zmid] - 0.5*dz + abs(r[i,2]-(z[zmid]-0.5*dz))
            r[i,6] = -r[i,6]
        while r[i,2] >= z[-1] + 0.5*dz:
            r[i,2] = z[-1]+ 0.5*dz - abs(r[i,2]-(z[-1]+0.5*dz))
            r[i,6] = -r[i,6]
    return r

def z_avg(arr):
    arr = np.sum(arr,axis=2)/arr.shape[2]
    arr = np.sum(arr,axis=1)/arr.shape[1]
    return arr

def Lorentzian(q,gamma):
    return gamma/(np.pi*(q**2 + gamma**2))

def BoltzmannDist(E_temp):
    return np.e**(-E_temp/kbT)

def E_thz(dtj_net,theta_i,theta_e):
    E_TE = np.zeros((theta_e.shape[0]))
    E_TM = np.zeros((theta_e.shape[0]))
    E_TMsin = np.zeros((theta_e.shape[0]))
    E_TMcos = np.zeros((theta_e.shape[0]))
                    
    te_coeff = 2*np.sin(theta_e)\
        /np.sin(theta_e + theta_i)
    E_TE = te_coeff*dtj_net[1]
    
    tm_coeff = 4*np.sin(theta_e)\
        /(np.sin(2*theta_e) + np.sin(2*theta_i))
    E_TM = tm_coeff*\
        (dtj_net[2]*np.sin(theta_i)-dtj_net[0]*np.cos(theta_i))
    E_TMsin = tm_coeff*\
        (dtj_net[2]*np.sin(theta_i))
    E_TMcos = tm_coeff*\
        (-dtj_net[0]*np.cos(theta_i))
                
    return E_TE,E_TM,E_TMsin,E_TMcos