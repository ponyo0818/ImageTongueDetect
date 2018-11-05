# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 11:40:27 2018
Train the classifier with data of color references and saved using pickle
@author: yfang
"""

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import ReferenceExtract
import pickle
#from sklearn.preprocessing import MinMaxScaler

def trainSVM():
    #CLASSIFIER   
    #Read the data and the label
    dfTrain=ReferenceExtract.referenceExtract()
    
    X = dfTrain.iloc[:,2:5] #pandas DataFrame
    y = dfTrain['label'] #pandas Series
    
    #split X and y into trainning and test sets

    #75%/25% train-test split
    X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=0)
    
# =============================================================================
#     
#     #Normalize the data with feature preprocessing using MinMax Scaling
#     scaler=MinMaxScaler()
#     X_train_scaled=scaler.fit_transform(X_train)
#     X_test_scaled=scaler.fit_transform(X_test)
# =============================================================================
    
   
# =============================================================================
#     #use the referenceExtract dataset to train 
#     #parameters select
#     parameters={'kernel':('linear','rbf'),'C':[0.001,0.01,0.1,1],'gamma':[0.001,0.01,0.1,1]}
#     #Using SVM classifier and fit the classifier with the train dataset
#     svc=svm.SVC() #trainning dataset accuracy:0.78, testset 0.77 gamma=0.01,c=0.1
#     clf=GridSearchCV(svc,parameters,cv=5)
#     print('Grid best parameter (max.accuracy):',grid_clf_acc.best_params_)
# =============================================================================
    
    clf=svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                decision_function_shape='ovr', degree=3, gamma='0.01', kernel='rbf',
                max_iter=-1, probability=False, random_state=None, shrinking=True,
                tol=0.001, verbose=False)
    clf.fit(X_train,y_train)
    
        #Find the accuracy of the svm classifier on trainning set
    print('Accuracy of SVC classifier on training set: {:.2f}'
         .format(clf.score(X_train,y_train)))
    
      #Find the accuracy of the svm classifier on test set
    print('Accuracy of SVC classifier on test set: {:.2f}'
         .format(clf.score(X_test,y_test)))
    
    
    print("Classifier:",clf)
    
    #Dump the trained classifier with Pickle
    model_pkl_filename='prediction_test_classifier.pkl'
    
    #Open the file to save as pkl file
    model_pkl=open(model_pkl_filename,'wb')
    pickle.dump(clf,model_pkl)
    
    #close the pickle instances
    model_pkl.close()
    
    #loading the saved prediction model
    model_pkl=open(model_pkl_filename,'rb')
    clf=pickle.load(model_pkl)
    return clf
    
    
    
    
    
    
   
    
    

    
    
    
    
    
    
    
    
    