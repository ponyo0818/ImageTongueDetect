
# -*- coding: utf-8 -*-
"""
This script is used to indentify the color based on rgb using svm in scikit-learn
Spyder Editor
Extract the test data from the images of Tongues
Identify the color of Tongues
yfang
"""

from collections import Counter
import pandas as pd


def detection(df,clf):
     #Using the svm classifier to predict for the test set
    dfTest=df
    
    
    X_test = dfTest.iloc[:,2:5] #pandas DataFrame
    
    
    #Remove the duplicate rows from pandas DataFrame
    X_test_short=X_test.drop_duplicates(subset=['red','green','blue'])
    
    #print(X_test_short)
    
    #predict the color of DataFrame been removed duplicate rows, in order to speed things up
    prediction_test=clf.predict(X_test_short) 
    #print(prediction_test)
    
    #Merge the prediction with the DataFrame
    X_test_short['preds']=prediction_test
    
    #print(X_test_short)
    #print(type(X_test_short))
    #print(type(X_test))
    #Merge it with the orgin dataframe
    df_out=pd.merge(X_test, X_test_short,how='left',on=['red','green', 'blue'])
    
    #print(df_out)

       
    # 舌苔1,2,3,4,5,6

    tonguecoat_name = ["danbai", "danhuang", "huang", "jiaohuang", "huihei","jiaohei", "bobai"]
    
    # 舌体 7,8,9,10,11

    tonguebody_name = ["danhong", "hong", "jiang", "qingzi", "bai"]
    
    #反光12 “fanguang"
    #special_name=["fanguang"]
    
    #convert the Numpy array to list
    #print(prediction_test)
    predictionList=df_out['preds'].tolist()
    #print(predictionList)
    
    #subset the colors to coat and body use list comprehension
    coatRows=[k for k in predictionList if k in tonguecoat_name]
    bodyRows=[k for k in predictionList if k in tonguebody_name]
    #print(coatRows,bodyRows)
    
    #Find the most common color in all the pixels of a image
    a=Counter(coatRows)
    b=Counter(bodyRows)
    #print(a,b)
    return a.most_common(5),b.most_common(5)
    








    
    
    
    