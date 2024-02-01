# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 22:58:22 2020

@author: Cerx
"""

import numpy as np
from scipy.constants import e, epsilon_0, m_e, Boltzmann
import sys
import os

wavelength = 800e-9 # E_photon = 1.55 eV
n = 3.5
I0 = int(7000) # cold_e = 7425
photon_number = np.copy(I0)
epsilon = epsilon_0*n**2

c = 3e8
dt = 2e-15
t_f = np.arange(0,101,1)*dt
v = 1*c/n
tR = 3e-6 #for n-type with doping ~1e14 cm^-3 / 1e20 m^-3
# tR = 2.5e-12 #2.5ps

alpha = 1.2e6 #interface depth = 1/alpha
xwidth = 3*1e-6
ywidth = 3*1e-6
zmaterialwidth = 4/alpha
zbeamwidth = c*100e-15
sigma_x = xwidth/20
sigma_y = xwidth/20
sigma_z = zbeamwidth/2
xlen,ylen,zlen = 15,11,30 #x & y must be odd, z must be even
zmid = int((zlen)/2)
ymid = int((ylen-1)/2)
dx = xwidth/(xlen)
dy = ywidth/(ylen)
dzmaterial = zmaterialwidth/(zlen)
dzbeam = zbeamwidth/(zlen)
x = np.linspace(-0.5*xwidth+0.5*dx,0.5*xwidth-0.5*dx,xlen)
y = np.linspace(-0.5*ywidth+0.5*dy,0.5*ywidth-0.5*dy,ylen)
Z = np.linspace(-0.5*zmaterialwidth,0.5*zmaterialwidth,zlen+1)
zmaterial = np.linspace(-0.5*zmaterialwidth+0.5*dzmaterial,\
                        0.5*zmaterialwidth-0.5*dzmaterial,zlen)
zbeam = np.linspace(-0.5*zbeamwidth+0.5*dzbeam,\
                    0.5*zbeamwidth-0.5*dzbeam,zlen)



Num_e = 2
num_e = xlen*ylen*(zmid)*Num_e
Num_h = 0
num_h = xlen*ylen*(zmid)*Num_h
volume = xwidth*ywidth*0.5*zmaterialwidth
donor = num_e/volume
doping = num_e/volume

E_photon = 1.55 #eV
E_bg = 1.5 #eV
phi_s = 0.5*E_bg
kbT = Boltzmann*300
me = 0.067*m_e
mh = 0.5*m_e
ve = np.sqrt(2*(E_photon-E_bg)*e/(me**-1 + mh**-1))/me
vh = np.sqrt(2*(E_photon-E_bg)*e/(me**-1 + mh**-1))/mh
ve_therm = np.sqrt(3*kbT/me)
mu_e = 8500 / 10000
mu_h = 400 / 10000
D_e = 200 / 10000
D_h = 10 / 10000

counter = 0
counter_end = 200 #(zwidth + 2/alpha)/(v*dt)
dtau = 2e-15

