# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 08:27:24 2018
Extract the information (r,g,b) from the color references
@author: yfang
"""


from PIL import Image
import os
import numpy as np
import pandas as pd


def referenceExtract():
    
    #EXTRACT THE DATA AND LABEL FROM COLOR REFERENCE
    
    data_path = 'C:\\Users\\FY\\Desktop\\TongueColorDetectYfang\\ColorReference'
    img_list = os.listdir(data_path)
    #img_list_nosuffix = os.path.splitext(data_path)[0]

    
    appendedDf=[] #create a empty dataframe
    for img_name in img_list:
        #Read the color and index
        colorImg=Image.open(os.path.join(data_path, img_name))    
        colorPixels=colorImg.convert("RGB")
        colorArray=np.array(colorPixels.getdata()).reshape(colorImg.size+(3,))
        indicesArray=np.moveaxis(np.indices(colorImg.size),0,2)
        #reshape the array 
        allArray=np.dstack((indicesArray,colorArray)).reshape((-1,5))
        
        df=pd.DataFrame(allArray,columns=["col","row","red","green","blue"])
        #Label each dataframe with the file name without suffix
        df["label"]=img_name[:-4]
        #print(df["label"])
        #append into a single dataframe
        appendedDf.append(df)
    appendedDf=pd.concat(appendedDf,axis=0) #concatenate the dataframes into a large dataframe
    return appendedDf

