# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 10:20:46 2018
Train the classifier with data of color references and saved using pickle
@author: yfang
"""

from sklearn import svm
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import GridSearchCV
import ReferenceExtract
#import pickle

from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score, classification_report
#from sklearn.preprocessing import MinMaxScaler


#Read the data and the label
dfTrain=ReferenceExtract.referenceExtract()

X = dfTrain.iloc[:,2:5] #pandas DataFrame
y = dfTrain['label'] #pandas Series

#split X and y into trainning and test sets

#75%/25% train-test split
X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=0)



#1.produces random prediction with same class proportion as training set
dummy_classprop=DummyClassifier(strategy='stratified').fit(X_train,y_train)
y_classprop_predicted=dummy_classprop.predict(X_test)
confusion=confusion_matrix(y_test,y_classprop_predicted)
print('Random class-proportional prediction(dummy classifier)\n',confusion)

#2 SVC classifier    

clf=svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
            decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
            max_iter=-1, probability=False, random_state=None, shrinking=True,
            tol=0.001, verbose=False)
clf.fit(X_train,y_train) 
svm_predicted=clf.predict(X_test)
confusion=confusion_matrix(y_test,svm_predicted)
print('Support Vector Machine classifier (rbf kernel)\n',confusion)

#3 Decision tree
dt=DecisionTreeClassifier(max_depth=2).fit(X_train,y_train)
tree_predicted=dt.predict(X_test)
confusion=confusion_matrix(y_test,tree_predicted)
print('Decision tree (max_depth=2)\n',confusion)



#Evaluation metrics for binary classification
#ramdom class-proportional dummy classifier
print('Random class-proportional (dummy)\n',
      classification_report(y_test,y_classprop_predicted))
print('Random class-proportional (dummy) Accuracy on test dataset: {:.2f}'.format(accuracy_score(y_test,y_classprop_predicted)))
print('train data: {:.2f}'
         .format(dummy_classprop.score(X_train,y_train)))
    
#SVM classifier
print('SVM\n',
      classification_report(y_test,svm_predicted))
print('SVM Accuracy: {:.2f}'.format(accuracy_score(y_test,svm_predicted)))
print('train data: {:.2f}'
         .format(clf.score(X_train,y_train)))

#Decision tree classifier
print('Decision tree',
      classification_report(y_test,tree_predicted))
print('Decision tree Accuracy: {:.2f}'.format(accuracy_score(y_test,tree_predicted)))
print('train data: {:.2f}'
         .format(dt.score(X_train,y_train)))






    
    
    
    
   
    
    

    
    
    
    
    
    
    
    
    