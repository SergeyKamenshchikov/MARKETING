#import libraries# 
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import library as mylib
from tabulate import tabulate
#/import libraries

#read xls into df#
path='DATA/report.xlsx';
df=pd.read_excel(path)
#/read xls into df#

#parameters#
pca=0
eda=0
#/parameters#
'''
Features: ['Data' 'Visits' 'Viewers'  'Views' 'BR' 'Depth' 'Time' 'Views/Viewer']
'''
#Print features,date#
print('----------------------------')
column_names=list(df.columns.values)
print('All fields:',column_names)
print('----------------------------')
print('Data Start:',list(df['Data'])[0])
print('Data End:',list(df['Data'])[-1])
print('Set power:', len(df))
print('----------------------------')
#/Print features and data#

#get/rescale exploratory features/target#
X_data=[]
X_data.extend([mylib.norm(list(df['Visits'])),mylib.norm(list(df['Viewers']))])
X_data.extend([mylib.norm(list(df['Views']))]),X_data.append(mylib.norm(list(df['Depth'])))
X_data.append(mylib.norm(mylib.timeconv(list(df['Time']))))
X_data.append(mylib.norm(list(np.array(list(df['Views']))/np.array(list(df['Viewers'])))))
column_names.append('Views/Viewer')
X_data=np.transpose(X_data)
del column_names[5];column_names=column_names[2:]

data_dict=dict(zip(column_names, X_data))
data_df=pd.DataFrame(X_data,columns=column_names)#features
fields=column_names
Target=mylib.norm(list(df['CR']));Target=np.transpose(Target)
print('Features:',fields)
#get/rescale exploratory features#

#exploratory analysis#
if pca==1:
    mylib.pca(column_names,data_df)
    print('----------------------------')

if eda==1:
    exp_df=data_df
    exp_df['CR']=np.transpose(mylib.norm(list(df['CR'])))
    mylib.cormap(exp_df)
    mylib.eda(exp_df)
    print('----------------------------')
    for item in range(0,len(fields)):
      print('Conversion vice '+str(fields[item])+'='+str(mylib.cf(X_data[:,item],Target)[0])+'%'+',p-significance='+str(mylib.cf(X_data[:,item],Target)[1]))
#/exploratory analysis#

'''
Features 1: ['Visits', 'Viewers', 'Views']
Features 2: ['Depth', 'Time', 'Views/Viewer']
'''
#split features#
print('----------------------------')
traffic=fields[0:3];print('Traffic features: '+str(traffic))
retention=fields[3:6];print('Retention features: '+str(retention))
df_traffic=pd.concat([data_df['Visits'], data_df['Viewers'], data_df['Views']], axis=1)#traffic
df_retention=pd.concat([data_df['Depth'], data_df['Time'], data_df['Views/Viewer']], axis=1)#retention
#/split features#

print('----------------------------')
s=[]
s.append(mylib.logreg(df_retention,Target)-mylib.logreg(df_traffic,Target))#logreg
s.append(mylib.knn(df_retention,Target)-mylib.knn(df_traffic,Target))#knn
s.append(mylib.tree(df_retention,Target)-mylib.tree(df_traffic,Target))#tree
s.append(mylib.mlp(df_retention,Target)-mylib.mlp(df_traffic,Target))#mlp
s=list(np.round(100*np.array(s))/100)

print('<Retention-Traffic, delta>:')
print('logreg_delta:'+str(s[0]))
print('knn_delta:'+str(s[1]))
print('tree_delta:'+str(s[2]))
print('mlp_delta:'+str(s[3]))
print('\naverage_delta:'+str(np.round(100*np.average(np.array(s)))/100))
print('std_delta:'+str(np.round(100*np.std(np.array(s)))/100))

print('----------------------------')

print('<Logreg>:')
print('mean_score_traffic:'+str(mylib.logreg(df_traffic,Target)))
print('mean_score_retention:'+str(mylib.logreg(df_retention,Target)))
print('----------------------------')
print('<Knn>:')
print('mean_score_traffic:'+str(mylib.knn(df_traffic,Target)))
print('mean_score_retention:'+str(mylib.knn(df_retention,Target)))
print('----------------------------')
print('<Tree>:')
print('mean_score_traffic:'+str(mylib.tree(df_traffic,Target)))
print('mean_score_retention:'+str(mylib.tree(df_retention,Target)))
print('----------------------------')
print('<MLP>:')
print('mean_score_traffic:'+str(mylib.mlp(df_traffic,Target)))
print('mean_score_retention:'+str(mylib.mlp(df_retention,Target)))

#FINISH#
print('----------------------------')
#/FINISH#

















