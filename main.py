# -*- coding: utf-8 -*-
"""
This script is used to indentify the color based on rgb using svm in scikit-learn
Spyder Editor
Read the color from reference
yfang yfang0818@gmail.com 2018/10/12
"""

from PIL import Image
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import svm


#EXTRACT THE DATA AND LABEL FROM COLOR REFERENCE

data_path = 'C:\\Users\\Administrator\\Desktop\\TongueColorDetectYfang\\ColorReference'
img_list = os.listdir(data_path)

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
    #Label each dataframe with the file name
    df["label"]=img_name
    #append into a single dataframe
    appendedDf.append(df)
appendedDf=pd.concat(appendedDf,axis=0) #concatenate the dataframes into a large dataframe


#CLASSIFIER
#Read the data and the lebel
X = appendedDf.iloc[:,2:5] #pandas DataFrame
y = appendedDf['label'] #pandas Series

#split X and y into trainning and test sets

#75%/25% train-test split
X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=0)

#Using SVM classifier and fit the classifier with the train dataset
clf=svm.SVC(gamma=0.001)
clf.fit(X,y)

#Using the svm classifier to predict for the test set
prediction_test=clf.predict(X_test)
print(prediction_test)
#Find the accuracy of the svm classifier using X_test and y_test
print('Accuracy of SVC classifier on test set: {:.2f}'
     .format(clf.score(X_test,y_test)))


#Find the accuracy of the svm classifier on trainning set
print('Accuracy of SVC classifier on training set: {:.2f}'
     .format(clf.score(X_train,y_train)))
