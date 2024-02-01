# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 22:58:22 2020

@author: Cerx
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from params3d import \
        xwidth, ywidth, zmaterialwidth, xlen, ylen, zlen, \
                ymid, zmid, dx, dy, dzmaterial, zmaterial, Num_e, Num_h

print('Time start:')
print(time.asctime())



                        
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