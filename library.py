import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests as gct
import scipy.stats
import re
from scipy.stats.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from statistics import median
from sklearn.model_selection import GridSearchCV
import warnings
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

#parameters#
print('----------------------------')
cv_n=10
out_sample=0.5
iterations=1000
neigbors_maximum=100
#parameters#

def parameters_out():
  print('cv_n='+str(cv_n))
  print('out_sample='+str(out_sample))
  print('iterations='+str(iterations))
  print('neigbors_maximum='+str(neigbors_maximum))

def adf(z):
  result=adfuller(z)
  delta=round(100*list(result[4].values())[1])/100-round(result[0])
  print('dADF Statistic for 5%:'+str(round(100*delta)/100))
  print('p-value: '+str(round(100*result[1]))+'%')

def eda(x):
  pd.plotting.scatter_matrix(x,figsize=[8,8],diagonal='kde',grid=True,ax=None,range_padding=1)
  plt.show()

def norm(x):
  x=np.array(x)-min(x)
  x=x/(max(x)-min(x))
  x=[float("{0:.2f}".format(item)) for item in list(x)]
  return x

def timeconv(x):
  sec=[];
  for item in x:
    h,m,s=re.split(':',item)
    total=int(h)*3600+int(m)*60+int(s)
    sec.append(total)
  return sec

def stand(x):
  x=np.array(x)
  x=x/np.std(x)
  return x  

def pca(x,y):
  print('----------------------------')
  pca=PCA()
  pca.fit(y)
  PCA(copy=True)
  features=x
  pca.explained_variance_=norm(pca.explained_variance_)
  pca.explained_variance_=list(np.array(pca.explained_variance_)*100)
  plt.bar(x, pca.explained_variance_,width=0.5)
  plt.xticks(features);plt.ylabel('variance');plt.xlabel('PCA feature');

  print('<PCA components>:')
  for i in range(0,len(pca.explained_variance_)):
    print(str(x[i])+': '+str(int(pca.explained_variance_[i]))+'%')
    i=i+1
  plt.show()

def gct(y,x):
  lag=1
  matrix=np.column_stack([y,x])
  gt=gct(matrix,3,verbose=False)
  gr_test=[item for item in gt.get(lag)[0]['params_ftest']];
  F_crit=int(round(scipy.stats.f.ppf(q=(1-gr_test[1]), dfn=gr_test[3], dfd=gr_test[2])))
  print('p='+str(format(gr_test[1], '.2f')))#p-value
  print("[F/F_crit]="+str(format(gr_test[0]/F_crit, '.2f')))#F/F_crit

def cf(x,y):
  CFV=pearsonr(x,y)
  correlation=str(abs(round(int(100*CFV[0]))))#correlation
  p_mystake=str(round((100*CFV[1]),2))#p-value
  cf_mst=[correlation,p_mystake]
  return cf_mst

def cormap(x):
  corr=x.corr()
  fig = plt.figure(figsize=(10, 7))
  g=sns.heatmap(corr,annot=True,linewidths=.2, cbar_kws={"orientation": "horizontal"})
  #g.set_xticklabels('labels', rotation=30)
  plt.show()

def get_class(x):
  threshold=median(list(x))
  class_type=[]
  for item in list(x):
    if item>=threshold: label=1;class_type.append(label)
    if item<threshold: label=0;class_type.append(label)
  return class_type  

def outlist(x):
  scores=stand(x)
  cv_score_mean=round(100*np.mean(x))/100;
  return cv_score_mean

def prediction(X_features,T,model):
  X_train,X_test,y_train,y_test=train_test_split(X_features,T,test_size=out_sample,random_state=21,stratify=T)
  model.fit(X_train,y_train)
  y_pred=list(model.predict(X_test))
  y_test=list(y_test)
  return [y_pred,y_test]

#############MODELS#############

##LOGREG##
def logreg(X_features,z):
  T1=get_class(z)

  scores=[]
  for item in range(2,cv_n):
    param_grid = {'tol': [0.1,0.01,0.001,0.0001]}
    logreg=LogisticRegression()
    logreg_cv=GridSearchCV(logreg, param_grid, scoring='roc_auc', cv=item)
    logreg_cv.fit(X_features,T1)
    tol_opt=logreg_cv.best_params_['tol']
    logreg_final=LogisticRegression(tol=tol_opt)
    cv_scores=list(cross_val_score(logreg_final,X_features,T1,cv=item,scoring='roc_auc'))

  return ([outlist(cv_scores)]+prediction(X_features,T1,logreg))

##FOREST##
def forest(X_features,z):
  T2=get_class(z)

  scores=[]
  for item in range(2,cv_n):
    forest=DecisionTreeClassifier(criterion='entropy')
    cv_scores=list(cross_val_score(forest,X_features,T2,cv=item,scoring='roc_auc'))

  return ([outlist(cv_scores)]+prediction(X_features,T2,forest))

##MLP##
def mlp(X_features,z):
  T3=get_class(z) 

  scores=[]
  for item in range(2,cv_n):
    clf_final=MLPClassifier(activation='relu', hidden_layer_sizes=(1,1), max_iter=iterations, random_state=1, solver='lbfgs')
    cv_scores=list(cross_val_score(clf_final,X_features,T3,cv=item,scoring='roc_auc'))

  return ([outlist(cv_scores)]+prediction(X_features,T3,clf_final))

##KNN##
def knn(X_features,z):
  T4=get_class(z)

  scores=[]
  for item in range(2,cv_n):
    param_grid = {'n_neighbors': np.arange(2,neigbors_maximum)}
    knn=KNeighborsClassifier()
    knn_cv=GridSearchCV(knn, param_grid, scoring='roc_auc', cv=item)
    knn_cv.fit(X_features,T4)
    n_opt=knn_cv.best_params_['n_neighbors']
    knn_final=KNeighborsClassifier(n_neighbors=n_opt)

    cv_scores=list(cross_val_score(knn_final,X_features,T4,cv=item,scoring='roc_auc'))
    scores.extend(cv_scores)

  return ([outlist(cv_scores)]+prediction(X_features,T4,knn_final))

##GNB##
def naivebayes(X_features,z):
  T5=get_class(z)

  scores=[]
  for item in range(2,cv_n):
    gnb=GaussianNB()
    cv_scores=list(cross_val_score(gnb,X_features,T5,cv=item,scoring='roc_auc'))
    scores.extend(cv_scores)

  return ([outlist(cv_scores)]+prediction(X_features,T5,gnb))
#############/MODELS#############

