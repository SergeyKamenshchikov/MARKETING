'''
Input data: #visitors,#views,#brate,#depth,#time,#sessions_per_visitor,#targets (contacts views) 

'''
#supress warning#OK
import warnings
warnings.filterwarnings("ignore")
#supress warning#

#import libraries#OK
import numpy as np
import pandas as pd
import re
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import FactorAnalysisSK
#/import libraries

#parameters#OK
show_adf=1
show_gct=1
show_corr=1
logreg=1
knn=1
show_corr_map=0
show_pca=1
#/parameters#

#traffic#[S1]OK
visitors=[float(line.strip()) for line in open('DATA/visitors.txt')]#visitors
views=[float(line.strip()) for line in open('DATA/views.txt')]#views
#traffic#

#involvment#[S1]OK
brate=[float(line.strip()) for line in open('DATA/brate.txt')]#brate
depth=[float(line.strip()) for line in open('DATA/depth.txt')]#depth
time=[str(line[3:].strip()) for line in open('DATA/time.txt')]#time
#involvment#

#time conversion#OK
time_upd=[];
for item in time:
  m,s=re.split(':',item)
  total_sec=float(m)*60+float(s)
  time_upd.append(total_sec)
time=time_upd
#time conversion#

#loyality#[S2]OK
sessions_per_visitor=[float(line.strip()) for line in open('DATA/sess.visitor.txt')]
#/loyality#

#targets#OK
targets=[float(line.strip()) for line in open('DATA/contacts.txt')]
#/targets#

#visitors,#views,#brate,#depth,#time,#sessions_per_visitor,#targets 

#rescaling#OK
visitors_sc=FactorAnalysisSK.NORM(visitors);views_sc=FactorAnalysisSK.NORM(views);
brate_sc=FactorAnalysisSK.NORM(brate);depth_sc=FactorAnalysisSK.NORM(depth);
time_sc=FactorAnalysisSK.NORM(time);sessions_per_visitor_sc=FactorAnalysisSK.NORM(sessions_per_visitor);
targets_sc=FactorAnalysisSK.NORM(targets);
#/rescaling#

#differentials#OK
diff_visitors=np.diff(visitors_sc);diff_views=np.diff(views_sc);
diff_brate=np.diff(brate_sc);diff_depth=np.diff(depth_sc);diff_time=np.diff(time_sc)
diff_sessions_per_visitor=np.diff(sessions_per_visitor_sc);
diff_targets=np.diff(targets_sc);
#/differentials#
 
#ADF#OK
if show_adf==1:
  print('\n\nResults of ADF test:') 
  features_adf={'diff_visitors':diff_visitors,'diff_views':diff_views,'diff_brate':diff_brate, 'diff_depth':diff_depth, 'diff_time':diff_time, 'diff_sessions_per_visitor':diff_sessions_per_visitor, 'diff_targets':diff_targets}
  for key in features_adf:
    print('-------------------')
    print('ADF:'+key)
    FactorAnalysisSK.ADF(features_adf[key])
#/ADF#
    
#GCT#OK
if show_gct==1:
  print('\n\nResults of GCT test:')   
  features_gct={'diff_visitors':diff_visitors,'diff_views':diff_views,'diff_brate':diff_brate, 'diff_depth':diff_depth, 'diff_time':diff_time, 'diff_sessions_per_visitor':diff_sessions_per_visitor} 
  for key in features_gct:
    print('-------------------')
    print('diff_targets('+key+')')
    FactorAnalysisSK.GCT(diff_targets,features_gct[key])
#/GCT#

targets

#CFT#OK
if show_corr==1:
  print('\n----------------------------------------')
  print('Results of Abs. Correlation test:')
  features_cft={'visitors_sc':time_sc, 'views_sc':views_sc, 'brate_sc':brate_sc, 'depth_sc':depth_sc, 'time_sc':time_sc, 'sessions_per_visitor_sc':sessions_per_visitor_sc}
  for key in features_cft:
    print('Correlation [targets_sc vs '+key+']'+'='+FactorAnalysisSK.CF(targets_sc,features_cft[key])[0]+'%'+'['+FactorAnalysisSK.CF(targets_sc,features_cft[key])[1]+'%]')
#/CFT#

#visitors,#views,#brate,#depth,#time,#sessions_per_visitor,#targets 

#PCA#OK 
column_names=['visitors','views','brate','depth','time','sessions_per_visitor','targets']
df_pca=pd.DataFrame(list(zip(visitors,views,brate,depth,time,sessions_per_visitor,targets)),columns=column_names)

if show_pca==1:
  pca=PCA();pca.fit(df_pca)
  PCA(copy=True)
  features=column_names
  plt.bar(features, pca.explained_variance_,width=0.5)
  plt.xticks(features);plt.ylabel('variance');plt.xlabel('PCA feature');
  plt.show()
#/PCA#

#LOGREG FA#OK
if logreg==1:
  print('\n----------------------------------------')
  print('Logreg for <views_sc> <views_sc>:')
  FactorAnalysisSK.LOGREGR(views_sc,views_sc,targets_sc)
  print('Logreg for <visitors_sc> <visitors_sc>:')
  FactorAnalysisSK.LOGREGR(visitors_sc,visitors_sc,targets_sc)
  print('Logreg for <visitors_sc> <views_sc>:')
  FactorAnalysisSK.LOGREGR(visitors_sc,visitors_sc,targets_sc)
  print('Logreg for <brate_sc> <brate_sc>:')
  FactorAnalysisSK.LOGREGR(brate_sc,brate_sc,targets_sc)
  print('Logreg for <depth_sc> <depth_sc>:')
  FactorAnalysisSK.LOGREGR(depth_sc,depth_sc,targets_sc)
  print('Logreg for <time_sc> <time_sc>:')
  FactorAnalysisSK.LOGREGR(time_sc,time_sc,targets_sc)
  print('Logreg for <sessions_per_visitor> <sessions_per_visitor>:')
  FactorAnalysisSK.LOGREGR(sessions_per_visitor_sc,sessions_per_visitor_sc,targets_sc)
#/LOGREG FA#

#KNN FA#OK
if knn==1:
  print('\n----------------------------------------')
  print('Number of neighbors are optimized 1-100')
  print('KNN for <views_sc> <views_sc>:')
  FactorAnalysisSK.KNN(views_sc,views_sc,targets_sc)
  print('KNN for <visitors_sc> <visitors_sc>:')
  FactorAnalysisSK.KNN(visitors_sc,visitors_sc,targets_sc)
  print('KNN for <visitors_sc> <views_sc>:')
  FactorAnalysisSK.KNN(visitors_sc,views_sc,targets_sc)
  print('KNN for <brate_sc> <brate_sc>:')
  FactorAnalysisSK.KNN(brate_sc,brate_sc,targets_sc)
  print('KNN for <depth_sc> <depth_sc>:')
  FactorAnalysisSK.KNN(depth_sc,depth_sc,targets_sc)
  print('KNN for <time_sc> <time_sc>:')
  FactorAnalysisSK.KNN(time_sc,time_sc,targets_sc)
  print('KNN for <sessions_per_visitor> <sessions_per_visitor>:')
  FactorAnalysisSK.KNN(sessions_per_visitor_sc,sessions_per_visitor_sc,targets_sc)
#/KNN FA#    

#corrmap#OK
if show_corr_map==1:
  corr=df_pca.corr();sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns)
  plt.suptitle('Correlations of rescaled')
  plt.show()
#/corrmap#

#corr#OK
if show_corr==1:
  print('\n----------------------------------------')
  print('Results of correlation test:')
  print('Correlation [visitors_sc vs '+'views_sc]'+'='+FactorAnalysisSK.CF(depth_sc,time_sc)[0]+'%'+','+'['+FactorAnalysisSK.CF(depth_sc,time_sc)[1]+'%]')
#/corr#

###filter dublicates#OK
##column_names=['visitors','views', 'brate','depth','time', 'sessions_per_visitor','targets']
##df_raw=pd.DataFrame(list(zip(visitors,views, brate,depth, time, sessions_per_visitor, targets)),columns=column_names).drop_duplicates(subset='targets',keep=False, inplace=False)
##visitors=list(df_raw['visitors'].values);views=list(df_raw['views'].values);
##brate=list(df_raw['brate'].values);depth=list(df_raw['depth'].values);time=list(df_raw['time'].values)
##sessions_per_visitor=list(df_raw['sessions_per_visitor'].values)
##targets=list(df_raw['targets'].values);
###/filter dublicates#
##
###rescaling#OK
##visitors_sc=FactorAnalysisSK.NORM(visitors);views_sc=FactorAnalysisSK.NORM(views);
##brate_sc=FactorAnalysisSK.NORM(brate);depth_sc=FactorAnalysisSK.NORM(depth);
##time_sc=FactorAnalysisSK.NORM(time);sessions_per_visitor_sc=FactorAnalysisSK.NORM(sessions_per_visitor);
##targets_sc=FactorAnalysisSK.NORM(targets);
#/rescaling#

#FINISH#OK
print('\nFinished!')
#/FINISH#

















