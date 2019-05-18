﻿#import libraries# 
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import library as mylib
from tabulate import tabulate
from datetime import datetime
#/import libraries

#read xls into df#
path='DATA/offers.xlsx';
df=pd.read_excel(path)
#/read xls into df#

###parameters#
##pca=1
##eda=1
##voting=3
###/parameters#
'''
Features: ['Data' 'Visits' 'Viewers'  'Views' 'BR' 'Depth' 'Time' 'Views/Viewer']
'''
#dates and offers from xls#
string_date=[]
offers_xls=[]
date_format = "%m/%d/%Y"
for i in range(len(df['Data'])):
    string_date.append(str(datetime.date(df['Data'][i])))
    offers_xls.append(int(df['Offers'][i]))
#/dates and offers from xls#    

#data benchmark#
data_range=pd.date_range(start='2016-05-09', end='2019-05-07')
dates=data_range.to_pydatetime();date_bench=[]
for item in dates: date_bench.append(item.strftime('%Y-%m-%d'))
#/data benchmark#

#fill spaces#
Offers=[]
for i in date_bench:#bench
    found=False
    for j in string_date:#xls
        if j==i:
            found=True
            break
    if found==True: Offers.append(0)
    if found==False: Offers.append(1) 
#/fill spaces#

#datetime.strptime(str(df['Data']), date_format)

###Print parameters#
##mylib.parameters_out()
###/Print parameters#
##
###Print features,date#
##print('----------------------------')
##column_names=list(df.columns.values)
##print('All fields:',column_names)
##print('----------------------------')
##print('Data Start:',list(df['Data'])[0])
##print('Data End:',list(df['Data'])[-1])
##print('Set power:', len(df))
##print('----------------------------')
###/Print features and data#
##
###get/rescale exploratory features/target#
##X_data=[]
##X_data.extend([mylib.norm(list(df['Visits'])),mylib.norm(list(df['Viewers']))])
##X_data.extend([mylib.norm(list(df['Views']))]),X_data.append(mylib.norm(list(df['Depth'])))
##X_data.append(mylib.norm(list(df['Time'])))
##X_data.append(mylib.norm(list(np.array(list(df['Views']))/np.array(list(df['Viewers'])))))
##column_names.append('Views/Viewer')
##X_data=np.transpose(X_data)
##del column_names[5];column_names=column_names[2:]
##
##data_dict=dict(zip(column_names, X_data))
##data_df=pd.DataFrame(X_data,columns=column_names)#features
##fields=column_names
##Target=mylib.norm(list(df['CR']));Target=np.transpose(Target)
##print('Features:',fields)
###get/rescale exploratory features#
##
###exploratory analysis#
##if pca==1:
##    mylib.pca(column_names,data_df)
##
##if eda==1:
##    exp_df=data_df
##    exp_df['CR']=np.transpose(mylib.norm(list(df['CR'])))
##    mylib.cormap(exp_df)
##    mylib.eda(exp_df)
##    print('----------------------------')
##    for item in range(0,len(fields)):
##      print('Conversion vice '+str(fields[item])+'='+str(mylib.cf(X_data[:,item],Target)[0])+'%'+',p-significance='+str(mylib.cf(X_data[:,item],Target)[1]))
###/exploratory analysis#
##
##'''
##Features 1: ['Visits', 'Viewers', 'Views']
##Features 2: ['Depth', 'Time', 'Views/Viewer']
##'''
##
###split features#
##print('----------------------------')
##traffic=fields[0:3];print('Traffic features: '+str(traffic))
##retention=fields[3:6];print('Retention features: '+str(retention))
##df_traffic=pd.concat([data_df['Visits'], data_df['Viewers'], data_df['Views']], axis=1)#traffic
##df_retention=pd.concat([data_df['Depth'], data_df['Time'], data_df['Views/Viewer']], axis=1)#retention
###/split features#
##
###absolute scores#
##print('----------------------------')
##print('<Logreg>:')
##print('mean_score_traffic:'+str(mylib.logreg(df_traffic,Target)[0]))
##print('mean_score_retention:'+str(mylib.logreg(df_retention,Target)[0]))
##
##print('----------------------------')
##print('<Knn>:')
##print('mean_score_traffic:'+str(mylib.knn(df_traffic,Target)[0]))
##print('mean_score_retention:'+str(mylib.knn(df_retention,Target)[0]))
##
##print('----------------------------')
##print('<Naive Bayes>:')
##print('mean_score_traffic:'+str(mylib.naivebayes(df_traffic,Target)[0]))
##print('mean_score_retention:'+str(mylib.naivebayes(df_retention,Target)[0]))
##
##print('----------------------------')
##print('<Decision tree>:')
##print('mean_score_traffic:'+str(mylib.dtree(df_traffic,Target)[0]))
##print('mean_score_retention:'+str(mylib.dtree(df_retention,Target)[0]))
##
##print('----------------------------')
##print('<MLP>:')
##print('mean_score_traffic:'+str(mylib.mlp(df_traffic,Target)[0]))
##print('mean_score_retention:'+str(mylib.mlp(df_retention,Target)[0]))
###/absolute scores#
##
###delta scores#
##print('----------------------------')
##s=[]
##s.append(mylib.logreg(df_retention,Target)[0]-mylib.logreg(df_traffic,Target)[0])#logreg
##s.append(mylib.knn(df_retention,Target)[0]-mylib.knn(df_traffic,Target)[0])#knn
##s.append(mylib.dtree(df_retention,Target)[0]-mylib.dtree(df_traffic,Target)[0])#forest
##s.append(mylib.naivebayes(df_retention,Target)[0]-mylib.naivebayes(df_traffic,Target)[0])#nbayes
##s.append(mylib.mlp(df_retention,Target)[0]-mylib.mlp(df_traffic,Target)[0])#mlp
##
##s=list(np.round(100*np.array(s))/100)
##
##print('<Retention-Traffic, delta>:')
##print('logreg_delta:'+str(s[0]))
##print('knn_delta:'+str(s[1]))
##print('dtree_delta:'+str(s[2]))
##print('nb_delta:'+str(s[3]))
##print('mlp_delta:'+str(s[4]))
##print('\naverage_delta:'+str(np.round(100*np.average(np.array(s)))/100))
##print('std_delta:'+str(np.round(100*np.std(np.array(s)))/100))
###/delta scores#
##
###ensemble scores for retention#
##print('----------------------------')
##a=mylib.logreg(df_retention,Target)[1]
##b=mylib.knn(df_retention,Target)[1]
##c=mylib.dtree(df_retention,Target)[1]
##d=mylib.naivebayes(df_retention,Target)[1]
##e=mylib.mlp(df_retention,Target)[1]
##
##test_list=mylib.logreg(df_retention,Target)[2]
##ensemble_score=[]
##for item in zip(a,b,c,d,e):
##    if sum(item)>=voting: ensemble_score.append(1)
##    if sum(item)<voting: ensemble_score.append(0)
##
##m=0;n=0
##for i in range(len(test_list)):
##    m=m+1;
##    if ensemble_score[i]==test_list[i]:n=n+1
##
##print('Retention ensemble accuracy:'+str(round(100*n/m))+'%')
###/ensemble scores for retention#
##
###ensemble scores for traffic#
##print('----------------------------')
##a=mylib.logreg(df_traffic,Target)[1]
##b=mylib.knn(df_traffic,Target)[1]
##c=mylib.dtree(df_traffic,Target)[1]
##d=mylib.naivebayes(df_traffic,Target)[1]
##e=mylib.mlp(df_traffic,Target)[1]
##
##test_list=mylib.logreg(df_traffic,Target)[2]
##ensemble_score=[]
##for item in zip(a,b,c,d,e):
##    if sum(item)>=voting: ensemble_score.append(1)
##    if sum(item)<voting: ensemble_score.append(0)
##
##m=0;n=0
##for i in range(len(test_list)):
##    m=m+1;
##    if ensemble_score[i]==test_list[i]:n=n+1
##
##print('Traffic ensemble accuracy:'+str(round(100*n/m))+'%')
###/ensemble scores for traffic#

#FINISH#
##print('----------------------------')
print('Finished')
#/FINISH#

















