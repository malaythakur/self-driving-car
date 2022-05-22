#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#reading images and steering angles from driving_Dataset folder

from __future__ import division

import os 
import numpy as np
import random

from scipy import pi
from itertools import islice


data_folder = 'D:/self-driving-car/Autopilot-TensorFlow-master/driving_dataset/' #driving_dataset location 
train_file = os.path.join(data_folder,'data.txt')

LIMIT = None

split = 0.8
x = []
y = []

with open(train_file) as fp:
    for line in islice(fp,LIMIT):
        path, angle = line.strip().split()
        full_path = os.path.join(data_folder, path)
        x.append(full_path)
        
        #converting angle to radians
        y.append(float(angle)* pi / 180)
        

y = np.array(y)
print('completed processing data.txt')

split_index = int(len(y)*0.8)

train_y = y[:split_index]
test_y = y[split_index:]

