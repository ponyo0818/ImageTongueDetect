# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 11:16:17 2018
Remove the noise (reduplicated pixels) from each referencing colors
Meanshift clustering in Scikit-learn
@author: yfang
"""
data_path = 'C:\\Users\\Administrator\\Desktop\\TongueColorDetectYfang\\ColorReference'

import numpy as np
import os
from sklearn.cluster import MeanShift,estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
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

plt.imshow(image)


#Estimate the kernel bandwidth to use from our image's datapoints
bandwidth=estimate_bandwidth(X,quantile=0.1,n_samples=100)
print(bandwidth)



#Run meanshif on the image to do the image segmentation, which stored in X
ms=MeanShift(bandwidth=bandwidth,bin_seeding=True)
ms.fit(X)

#print some related information
labels=ms.labels_
print(labels.shape)
cluster_centers=ms.cluster_centers_
print(cluster_centers.shape)

labels_unique=np.unique(labels)
n_clusters=len(labels_unique)

print("number of estimated clusters : %d" % n_clusters)

segmented_image=np.reshape(labels,original_shape[:2]) #just take the size, ignore the RGB channels

#display the segmented image and the original image
plt.figure(2)
plt.subplot(1,2,1)
plt.imshow(image)
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(segmented_image)
plt.axis('off')





