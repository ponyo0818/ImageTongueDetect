# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 08:34:02 2018
SVM Identify the color of Tongues

@author: yfang  yfang0818@gmail.com 
"""


from PIL import Image
import os
import numpy as np
import pandas as pd
import TongueDetect
import TrainSVM
import time


#EXTRACT THE DATA AND LABEL FROM TONGUE IMAGES

data_path1 = 'C:\\Users\\FY\\Desktop\\TongueColorDetectYfang\\Tongues\\resize'
img_list1 = os.listdir(data_path1)
#img_list1_nosuffix = os.path.splitext(data_path1)[0]


#Train the model using data of color references
clf=TrainSVM.trainSVM()  
 
#Predict the color of each tongue images
for img_name in img_list1:
    #Read the color and index
    print('Begin to analysis the color of image',img_name)
    print (time.localtime(time.time()))
    colorImg=Image.open(os.path.join(data_path1, img_name))    
    colorPixels=colorImg.convert("RGB")
    colorArray=np.array(colorPixels.getdata()).reshape(colorImg.size+(3,))
    indicesArray=np.moveaxis(np.indices(colorImg.size),0,2)
    #reshape the array 
    allArray=np.dstack((indicesArray,colorArray)).reshape((-1,5))
    
    df=pd.DataFrame(allArray,columns=["col","row","red","green","blue"])
    #Label each dataframe with the file name
    df["number"]=img_name

    coat,body=TongueDetect.detection(df,clf)
    print("舌质和舌苔的颜色分别是",body,coat)


   

