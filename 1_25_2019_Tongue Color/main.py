# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 08:56:39 2019
To discover the patterns in the data of TongueColor
and make prediction of the color of Tongue using ML models
of linearRegression, SVM, and Xgbox.


鉴定结果（Label）	
淡红	danhong
红	hong
绛	jiang

光照(brightness)	
半暗半亮	0
光照正常	1
曝光过度	2
整体偏暗	3


@author: FY
"""
import pandas as pd
import numpy as np
import sklearn
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model

def main():
    dataPath='C:\\Users\\FY\\Desktop\\TongueColorDetectYfang\\1_25_2019_Tongue Color\\TongueColor.csv'
    tongueColor=pd.read_csv(dataPath,sep=",",nrows=4123) #read csv file into pandas dataframe    
    print (tongueColor)
    print(tongueColor.dtypes)
    DataSummary(tongueColor)
    LogReg(tongueColor)
    SupportVectorMachine(tongueColor)
    
    
def DataSummary(tongueColor):
    #count the number of each type
    print(tongueColor['label'].value_counts())
    
    #visualization
    sns.countplot(x='label',data=tongueColor,palette='hls')
    plt.savefig('data summary.png')
    plt.show()


"""Accuracy of SVC classifier on test set: 0.90
Accuracy of SVC classifier on training set: 0.93"""
def LogReg(tongueColor):
    #Read the data and the lebel
    X = tongueColor.loc[:,'danbai':'danzi'] #pandas DataFrame
    y = tongueColor['label'] #pandas Series
    
    #split X and y into trainning and test sets
    
    #75%/25% train-test split
    X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=0)
    
    #Create the logisticRegression object
    logreg=linear_model.LogisticRegression(solver='lbfgs',C=1e5)
    logreg.fit(X_train,y_train)
    print(logreg)
    #predict the test set result and calculating the accuracy
    y_pred=logreg.predict(X_test)
    print('number of sample been predicted: ',len(y_pred))
    print ('Accuray of logistic regression classifier on test set:{:.2f}'.format(logreg.score(X_test,y_test)))
    
    #Find the accuracy of the svm classifier using X_test and y_test
    print('Accuracy of SVC classifier on test set: {:.2f}'
         .format(logreg.score(X_test,y_test)))
    
    
    #Find the accuracy of the svm classifier on trainning set
    print('Accuracy of SVC classifier on training set: {:.2f}'
         .format(logreg.score(X_train,y_train)))
    
    
    #plot the scatter plot to compare true types and the predicted types.
    plt.scatter(y_test,logreg.predict(X_test),alpha=0.1,s=70)
    plt.xlabel("type: $Y_i$")
    plt.ylabel("Predicted tyep: $\hat{Y}_i$")
    plt.title("Logistic Regression types vs Predicted types:$Y_i$ vs $\hat{Y}_i$" )
    plt.savefig('logistic regression')


    
"""Accuracy of SVC classifier on test set: 0.79
Accuracy of SVC classifier on training set: 0.98"""
def SupportVectorMachine(tongueColor):
    #CLASSIFIER
    #Read the data and the lebel
    X = tongueColor.loc[:,'danbai':'danzi'] #pandas DataFrame
    y = tongueColor['label'] #pandas Series
    
    #split X and y into trainning and test sets
    
    #75%/25% train-test split
    X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=0)
    
    #Using SVM classifier and fit the classifier with the train dataset
    clf=svm.SVC()
    clf.fit(X_train,y_train)
    
    #Using the svm classifier to predict for the test set
    y_pred=clf.predict(X_test)
    print('number of sample been predicted: ',len(y_pred))

    #Find the accuracy of the svm classifier using X_test and y_test
    print('Accuracy of SVC classifier on test set: {:.2f}'
         .format(clf.score(X_test,y_test)))
    
    
    #Find the accuracy of the svm classifier on trainning set
    print('Accuracy of SVC classifier on training set: {:.2f}'
         .format(clf.score(X_train,y_train)))
    
    print('number of sample been predicted: ',len(y_pred))
    #plot the scatter plot to compare true types and the predicted types.
    plt.scatter(y_test,clf.predict(X_test),alpha=0.1,s=70)
    plt.xlabel("type: $Y_i$")
    plt.ylabel("Predicted tyep: $\hat{Y}_i$")
    plt.title("SVM types vs Predicted types:$Y_i$ vs $\hat{Y}_i$" )
    plt.savefig('svm')
    

# =============================================================================
# def Xgbox():
#     
# =============================================================================
    
if __name__ == "__main__":
    main()
    
