'''
Input data: Visitors,Views,BR,Depth,Sessions,Targets [views of contacts + content]
Data: 03-03-18/03-03-19 [1 year,days,363 records]
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
show_corr_map=0
show_corr=1
show_pca=0
#/parameters#

#traffic#OK
visitors=[float(line.strip()) for line in open('DATA/visitors.txt')]#visitors
views=[float(line.strip()) for line in open('DATA/views.txt')]#views
sessions=[float(line.strip()) for line in open('DATA/sessions.txt')]#sessions
#traffic#

#involvment#OK
brate=[float(line.strip()) for line in open('DATA/brate.txt')]#brate
depth=[float(line.strip()) for line in open('DATA/depth.txt')]#depth
time=[str(line[3:].strip()) for line in open('DATA/time.txt')]#time
#involvment#

#loyality#OK
sessions_per_visitor=[x/y for x, y in zip(sessions,visitors)]
#/loyality#

#time conversion#OK
time_upd=[];
for item in time:
  m,s=re.split(':',item)
  total_sec=float(m)*60+float(s)
  time_upd.append(total_sec)
time=time_upd
#time conversion#

#conversion#OK
targets=[float(line.strip()) for line in open('DATA/targets.txt')]
conversion=[x/y for x, y in zip(targets,views)]
#/covcersion#

#filter target nulls#OK
column_names=['visitors','views', 'brate','depth','time', 'sessions_per_visitor','conversion','targets']
df_raw=pd.DataFrame(list(zip(visitors,views, brate,depth, time, sessions_per_visitor, conversion, targets)),columns=column_names)
df_raw=df_raw[df_raw['targets']>0]

visitors=list(df_raw['visitors'].values);views=list(df_raw['views'].values);
brate=list(df_raw['brate'].values);depth=list(df_raw['depth'].values);time=list(df_raw['time'].values)
sessions_per_visitor=list(df_raw['sessions_per_visitor'].values)
conversion=list(df_raw['conversion'].values);
#/filter target nulls#

#rescaling#OK
visitors_sc=FactorAnalysisSK.NORM(visitors);views_sc=FactorAnalysisSK.NORM(views);
brate_sc=FactorAnalysisSK.NORM(brate);depth_sc=FactorAnalysisSK.NORM(depth);time_sc=FactorAnalysisSK.NORM(time)
sessions_per_visitor_sc=FactorAnalysisSK.NORM(sessions_per_visitor);
conversion_sc=FactorAnalysisSK.NORM(conversion);
#/rescaling#

#differentials#OK
diff_visitors=np.diff(visitors_sc);diff_views=np.diff(views_sc);
diff_brate=np.diff(brate_sc);diff_depth=np.diff(depth_sc);diff_time=np.diff(time_sc)
diff_sessions_per_visitor=np.diff(sessions_per_visitor_sc);
diff_conversion=np.diff(conversion_sc);
#/differentials#
 
#ADF#OK
if show_adf==1:
  print('\n\nResults of ADF test:') 
  features_adf={'diff_visitors':diff_visitors,'diff_views':diff_views,'diff_brate':diff_brate, 'diff_depth':diff_depth, 'diff_time':diff_time, 'diff_sessions_per_visitor':diff_sessions_per_visitor, 'diff_conversion':diff_conversion}
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
    print('diff_conversion('+key+')')
    FactorAnalysisSK.GCT(diff_conversion,features_gct[key])
#/GCT#

#CFT#OK
if show_corr==1:
  print('\n----------------------------------------')
  print('Results of Abs. Correlation test:')
  features_cft={'depth_sc':depth_sc, 'time_sc':time_sc}
  for key in features_cft:
    print('Correlation [conversion_sc vs '+key+']'+'='+FactorAnalysisSK.CF(conversion_sc,features_cft[key])[0]+'%'+'['+FactorAnalysisSK.CF(conversion_sc,features_cft[key])[1]+'%]')
#/CFT#

#PCA#OK 
column_names=['visitors_sc','views_sc','brate_sc','depth_sc','time_sc','sessions_per_visitor_sc','conversion_sc']
df_pca=pd.DataFrame(list(zip(visitors_sc,views_sc,brate_sc,depth_sc,time_sc,sessions_per_visitor_sc,conversion_sc)),columns=column_names)

if show_pca==1:
  pca=PCA();pca.fit(df_pca)
  PCA(copy=True)
  features=column_names
  plt.bar(features, pca.explained_variance_,width=0.5)
  plt.xticks(features);plt.ylabel('variance');plt.xlabel('PCA feature');
  plt.show()
#/PCA#

#LOGREG FA#OK
print('\n----------------------------------------')
print('Logreg for <depth_sc>:',)
FactorAnalysisSK.LOGREGR(depth_sc,depth_sc,conversion_sc)
print('Logreg for <time_sc>:')
FactorAnalysisSK.LOGREGR(time_sc,time_sc,conversion_sc)
#/LOGREG FA#    

#corrmap#OK
if show_corr_map==1:
  corr=df_pca.corr();sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns)
  plt.suptitle('Correlations of rescaled')
  plt.show()
#/corrmap#

#corr#OK
if show_corr==1:
  print('\n----------------------------------------')
  print('Results of logistic regression test:')
  print('Correlation [depth_sc vs '+'time_sc]'+'='+FactorAnalysisSK.CF(depth_sc,time_sc)[0]+'%'+','+'['+FactorAnalysisSK.CF(depth_sc,time_sc)[1]+'%]')
#/corr#    

#FINISH#OK
print('\nFinished!')
#/FINISH#

















