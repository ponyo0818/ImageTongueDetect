# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 11:16:17 2018
Remove the noise (reduplicated pixels) from each referencing colors
Fuzzy C-means clustering in Scikit-learn
@author: yfang
"""
data_path = 'C:\\Users\\FY\\Desktop\\TongueColorDetectYfang\\ColorReference'

import numpy as np
import os
from __future__ import division, print_function
import skfuzzy as fuzz
from sklearn.cluster import MeanShift,estimate_bandwidth
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
#%matplotlib inline
pylab. rcParams['figure.figsize']=10,10

image=Image.open(os.path.join(data_path, 'bai.png'))
image=np.array(image)
original_shape=image.shape

#flatten the image
X=np.reshape(image,[-1,3])


