# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 22:58:22 2020

@author: Cerx
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as img
import time


print('Time start:')
print(time.asctime())

frames = []
fig = plt.figure(figsize=(12,4),dpi=300,frameon=False,layout='tight')
for i in np.arange(0,1501,10):
    im_temp = plt.imshow(img.imread('thz_6_0_2 _ %s.png'%i),animated=True)
    plt.axis('off')
    frames.append([im_temp])
anim = animation.ArtistAnimation(fig, frames, interval=50, blit=True)
anim.save('carrier_drift.mp4')

print('Time start:')
print(time.asctime())