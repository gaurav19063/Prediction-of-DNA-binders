# -*- coding: utf-8 -*-
"""
Gaurav Lodhi 
MT19063
Note: There may have some problems in running the file because of local libraries. This is my humble request either update the libraries or use the code on google colab.

"""


# importing required packages
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import RandomOverSampler





# Amino acid composition for features
def get_features_AAC(df,list_peptides):
  rows=df.size  #no of rows
  df_f=pd.DataFrame(columns=list_peptides) #empty dataframe of list peptides
  for index in range(rows) : 
    df_f.loc[index]=0
    seq=df['Sequence'].loc[index]
    for i in range(len(seq)):
      col=seq[i]
      # print(col)
      df_f.loc[index][col]=df_f.loc[index][col]+1
    df_f.loc[index]=df_f.loc[index]/(len(seq))
  return  df_f


#  Dipeptide composition for features
def get_features_DP(df,d,list_peptides):
  rows=df.size  #no of rows
  df_f=pd.DataFrame(columns=list_peptides) #empty dataframe of list peptides
  for index in range(rows) : 
    df_f.loc[index]=0
    seq=df['Sequence'].loc[index]
    for i in range(len(seq)-d-1):
      col=seq[i]+seq[i+d+1]
      # print(col)
      df_f.loc[index][col]=df_f.loc[index][col]+1
    df_f.loc[index]=df_f.loc[index]/(len(seq)-(d+1))
  return  df_f


# Extracting Features from sequences 
def Feature_Extraction_Storing():
  print("Feature Extracting It may takes time if you want to avoid comment the method and use stored preprocessed data...")
  list_dfs_train=[]
  list_dfs_test=[]
  for d in range (6):
    list_dfs_train.append(get_features_DP(data_train,d,list_dipeptides))
    list_dfs_test.append(get_features_DP(data_test,d,list_dipeptides))
    print("Complete for ",d,"th order dipeptide")
  list_dfs_train.append(get_features_AAC(data_train,list_aminoAcids))
  list_dfs_test.append(get_features_AAC(data_test,list_aminoAcids))
  print("Feature Extraction Complete")
  train_data = pd.concat(list_dfs_train, axis=1, sort=False)
  test_data = pd.concat(list_dfs_test, axis=1, sort=False)

  # Storing the feature extracted dataset 
  train_data.to_pickle('train_ext_3.pkl')
  test_data.to_pickle('test_ext_3.pkl')
  print("Feature Extracting Complete and saved in a pickle file.")


# Cross validation for performance analysis.
def CV(model,X,y):
  cv_results = cross_val_score(model, X, y, cv=5)
  print(cv_results)
  print(np.mean(cv_results))


def main(path_train,path_test,output_file_name):
  # Reading the train.csv and valid.csv and seperating labels from the dataset
  data=pd.read_csv(path_train)
  data=data.rename(columns={' Sequence':'Sequence',' Type':'Type'})
  print(Counter(data['Type']))
  Y_train=data['Type']
  Y_train=Y_train.replace({'NDNA':-1,'DNA':1})
  data_train=data.drop(['ID','Type'],axis=1,inplace=False)

  data_test=pd.read_csv(path_test)
  data_test=data_test.rename(columns={' Sequence': 'Sequence'})
  ids=data_test['ID']
  data_test.drop(['ID'],axis=1,inplace=True)



  # making list of all the possible diapeptides and tripeptides.
  list_tripeptides=[]
  list_aminoAcids=['A','R','N','D','C','E','Q','G','H','I','L','K','M','F','P','S','T','W','Y','V','X']
  list_dipeptides=[]
  for x in list_aminoAcids:
      for y in list_aminoAcids:
          list_dipeptides.append(x+y)
          for z in list_aminoAcids:
              # print(x+y+z)
              list_tripeptides.append(x+y+z)
  print(len(list_tripeptides))
  print(len(list_dipeptides))


  # Feature Extraction it will take time to extract the features. To save some time comment it and use the prepared datastored in the pickle files.

  Feature_Extraction_Storing()


  # Read prepared stored data
  X_train=pd.read_pickle('train_ext_3.pkl',)
  X_test=pd.read_pickle('test_ext_3.pkl')
  #X_train = np.asarray(X_train).astype('float32')
  #X_test=np.asarray(X_test).astype('float32')

  # To check the information of the dataset
  # train.info()
  # test.info()

  # SelectKBest for feature selection
  print("Feature Selection is happening using SelectKBest.")

  select=SelectKBest(chi2, k=1900) 
  select.fit_transform(X_train, Y_train)
  cols = select.get_support(indices=True)
  X_train_selected = X_train.iloc[:,cols]
  X_test_selected=X_test.iloc[:,cols]

  # Using Oversampling to Balance the classes.

  oversample = RandomOverSampler(sampling_strategy='minority',random_state=48)
  # fit and apply the transform
  X_over, y_over = oversample.fit_resample(X_train_selected, Y_train)
  # summarize class distribution
  print(Counter(y_over))


  # find cross Validation of the model
  print("Cross Validation for Model Preformance Analysis.")
  c=10
  model=SVC(C= c, gamma= 1, kernel= 'rbf')
  CV(model,X_over,y_over)

  # GridSearchCV for Hyper parameter tunning.(commented because it takes time to process.)
  # print("Grid Search for hyper parameter tunning.")
  # param_grid = {'C':[1,10,100,1000],'gamma':[1,0.1,0.001,0.0001], 'kernel':['linear','rbf']}
  # grid = GridSearchCV(SVC(),param_grid,refit = True, verbose=2,cv=5, scoring='accuracy')
  # grid.fit(train,Y_train)
  # grid.best_params_


  # Model training
  print("Model Trainig...")
  # model=SVC(C= 10, gamma= 1, kernel= 'rbf')
  model.fit(X_over, y_over)

  # prediction of the model
  prediction=model.predict(X_test_selected)
  print(Counter(prediction))

  prediction=pd.DataFrame(data=prediction)
  new_dataframe=pd.DataFrame(columns=['ID','Lable'])
  new_dataframe['ID']=ids
  new_dataframe['Lable']=prediction
  new_dataframe=new_dataframe.set_index('ID')
  # print(new_dataframe)

  # Storing the result
  new_dataframe.to_csv(output_file_name)


########################################################################
# run code from here

path_train='train.csv'
path_test='valid.csv'
output_file_name='output.csv'

main(path_train,path_test,output_file_name)

########################################################################


