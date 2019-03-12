import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests as gct
import scipy.stats
from scipy.stats.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from statistics import median

def ADF(z):
  result=adfuller(z)
  print('ADF Statistic-Visitors: %f' % result[0]);print('p-value: %f' % result[1]);print('Critical Values:')
  for key, value in result[4].items():print('\t%s: %.3f' % (key, value))

def NORM(x):
  x=np.array(x)-min(x)
  x=x/(max(x)-min(x))
  return x

def STAND(x):
  x=np.array(x)-np.mean(x)
  x=x/np.std(x)
  return x  

def GCT(y,x):
  lag=1
  matrix=np.column_stack([y,x])
  gt=gct(matrix,3,verbose=False)
  gr_test=[item for item in gt.get(lag)[0]['params_ftest']];
  F_crit=int(round(scipy.stats.f.ppf(q=(1-gr_test[1]), dfn=gr_test[3], dfd=gr_test[2])))
  print('p='+str(format(gr_test[1], '.2f')))#p-value
  #print("[F/F_crit]="+str(format(gr_test[0]/F_crit, '.2f')))#F/F_crit

def CF(x,y):
  CFV=pearsonr(x,y)
  correlation=str(abs(round(int(100*CFV[0]))))#correlation
  p_mystake=str(round((100*CFV[1]),2))#p-value
  cf_mst=[correlation,p_mystake]
  return cf_mst

def LOGREGR(x,y,z):
  logreg=LogisticRegression()

  threshold=median(list(z)) 
  l=[]
  for item in list(z):
    if item>=threshold: label=1;l.append(label)
    if item<threshold: label=0;l.append(label)  

  x1=list(x);x2=list(y);
  X_features=np.array([x1,x2]);X_features=np.transpose(X_features)
  cv_scores=cross_val_score(logreg,X_features,l,cv=5,scoring='roc_auc')

  cv_score_mean=100*np.mean(cv_scores);cv_score_mean=cv_score_mean.astype(int)
  cv_score_std=100*np.std(cv_scores);cv_score_std=cv_score_std.astype(int)
  score_list=[cv_score_mean,cv_score_std] 
  print('<SCORE_INV='+str(np.around(score_list[0]))+'%'+'['+str(np.around(score_list[1]))+'%]')
