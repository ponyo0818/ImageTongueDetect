# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 08:56:39 2019
To discover the patterns in the data of TongueColor
and make prediction of the color of Tongue using ML models
of linearRegression, SVM, and Xgboost.


鉴定结果（Label）	
淡白 danbai
淡红	danhong
红	hong
绛	jiang
淡紫 danzi


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
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


def main():
    dataPath='C:\\Users\\FY\\Desktop\\TongueColorDetectYfang\\1_29_2019_TongueColor\\TongueColor_1_29_2019.csv'
    tongueColor=pd.read_csv(dataPath,sep=",",nrows=6170,encoding='unicode_escape') #read csv file into pandas dataframe    
    print (tongueColor)
    print(tongueColor.dtypes)
    DataSummary(tongueColor)
    LogReg(tongueColor)
    SupportVectorMachine(tongueColor)
    Xgboost(tongueColor)
    
def DataSummary(tongueColordata):
    #count the number of each type
    print(tongueColordata['label'].value_counts())
    
    #visualization
    sns.countplot(x='label',data=tongueColordata,palette='hls')
    plt.savefig('data summary.png')
    plt.show()


"""Accuracy of logistic Regression classifier on test set: 0.65
Accuracy of logistic Regression classifier on training set: 0.67"""
def LogReg(tongueColordata):
    tongueColor=tongueColordata
    #Read the data and the lebel
    X = tongueColor.loc[:,'danbai':'brightness'] #pandas DataFrame
    y = tongueColor['label'] #pandas Series
    
    #standardize the data
    scaler=StandardScaler().fit(X)
    rescaledX=scaler.transform(X)
    #split X and y into trainning and test sets
    
    #75%/25% train-test split
    X_train, X_test, y_train, y_test=train_test_split(rescaledX,y,random_state=0)
    
    #Create the logisticRegression object
    logreg=linear_model.LogisticRegression(solver='lbfgs',C=1e5)
    logreg.fit(X_train,y_train)
    print(logreg)
    #predict the test set result and calculating the accuracy
    y_pred=logreg.predict(X_test)
    
    #add prediction to the dataframe and write it in a csv file
    all_pred=logreg.predict(rescaledX)
    tongueColor["predict"]=all_pred
    print(tongueColor)
    tongueColor.to_csv('logReg.csv',sep=',',encoding='utf-8')
    
    
    print('number of sample been predicted: ',len(y_pred))
    print ('Accuray of logistic regression classifier on test set:{:.2f}'.format(logreg.score(X_test,y_test)))
    
    #Find the accuracy of the svm classifier using X_test and y_test
    print('Accuracy of logistic Regression classifier on test set: {:.2f}'
         .format(logreg.score(X_test,y_test)))
    
    
    #Find the accuracy of the svm classifier on trainning set
    print('Accuracy of logistic Regression classifier on training set: {:.2f}'
         .format(logreg.score(X_train,y_train)))
    
    
    #plot the scatter plot to compare true types and the predicted types.
    plt.scatter(y_test,logreg.predict(X_test),alpha=0.1,s=70)
    plt.xlabel("type: $Y_i$")
    plt.ylabel("Predicted tyep: $\hat{Y}_i$")
    plt.title("Logistic Regression types vs Predicted types:$Y_i$ vs $\hat{Y}_i$" )
    plt.savefig('logistic regression')


    
"""Accuracy of SVM classifier on test set: 0.65
Accuracy of SVM classifier on training set: 0.73"""
def SupportVectorMachine(tongueColordata):
    tongueColor=tongueColordata

    #CLASSIFIER
    #Read the data and the lebel
    X = tongueColor.loc[:,'danbai':'danzi'] #pandas DataFrame
    y = tongueColor['label'] #pandas Series
    

    #split X and y into trainning and test sets
    
    #75%/25% train-test split
    X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=0)
    
# =============================================================================
#     #set the parameters by cross-validation
#     tuned_parameters={'kernel':('linear','rbf'),'C':[10,100],'gamma':[0.001,0.01]}
#     
# 
#     print("#Tunning hyper-parameters for accuracy")
#     print()
#     
#     clf=GridSearchCV(SVC(),tuned_parameters,cv=5)
#     clf.fit(X_train,y_train)
#     print('Best parameters set found on development set:')
#     print(clf.best_params_)
#     
# =============================================================================
    #Using SVM classifier and fit the classifier with the train dataset
    clf=svm.SVC(gamma=0.01)
    clf.fit(X_train,y_train)
    print (clf)

    
    #Using the svm classifier to predict for the test set
    y_pred=clf.predict(X_test)
    print('number of sample been predicted: ',len(y_pred))
    
    #add prediction to the dataframe and write it in a csv file
    all_pred=clf.predict(X)
    tongueColor["predict"]=all_pred
    print(tongueColor)
    tongueColor.to_csv('SVM.csv',sep=',',encoding='utf-8')
    
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
    
"""Accuracy of Xgboost classifier on test set: 0.65
Accuracy of Xgboost classifier on training set: 0.70"""
def Xgboost(tongueColordata):
    tongueColor=tongueColordata

    #CLASSIFIER
    #Read the data and the lebel
    X = tongueColor.loc[:,'danbai':'danzi'] #pandas DataFrame
    y = tongueColor['label'] #pandas Series
    
    #split X and y into trainning and test sets
    
    #75%/25% train-test split
    X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=0)
    
    #Using SVM classifier and fit the classifier with the train dataset
    xb=XGBClassifier(learning_rate=0.1,gamma=0.1)
    xb.fit(X_train,y_train)
    print(xb)
    #Using the svm classifier to predict for the test set
    y_pred=xb.predict(X_test)
    print('number of sample been predicted: ',len(y_pred))
    
    #add prediction to the dataframe and write it in a csv file
    all_pred=xb.predict(X)
    tongueColor["predict"]=all_pred
    print(tongueColor)
    tongueColor.to_csv('xgboost.csv',sep=',',encoding='utf-8')
    #Find the accuracy of the svm classifier using X_test and y_test
    print('Accuracy of Xgboost classifier on test set: {:.2f}'
         .format(xb.score(X_test,y_test)))
    
    
    #Find the accuracy of the svm classifier on trainning set
    print('Accuracy of Xgboost classifier on training set: {:.2f}'
         .format(xb.score(X_train,y_train)))
    
    print('number of sample been predicted: ',len(y_pred))
    #plot the scatter plot to compare true types and the predicted types.
    plt.scatter(y_test,xb.predict(X_test),alpha=0.1,s=70)
    plt.xlabel("type: $Y_i$")
    plt.ylabel("Predicted tyep: $\hat{Y}_i$")
    plt.title("Xgboost types vs Predicted types:$Y_i$ vs $\hat{Y}_i$" )
    plt.savefig('xgboost')
    
    
if __name__ == "__main__":
    main()
    
