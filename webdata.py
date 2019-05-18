#libraries#
import requests
import pandas as pd
import json
import xlsxwriter
import os
#/libraries#

#parameters#
token='OAuth AQAAAAAi8O5WAAWWtZPDULmLHkZ8hRamVGxB9SM'
counter='29709450' #lab-test
counter_ym='44147844' #yandex
goal_hits='30606879'
goal_blog='30606884'
goal_features='30606889'#
goal_try='41646742'
#/parameters#

header = {'Authorization': 'OAuth AQAAAAAi8O5WAAWWtZPDULmLHkZ8hRamVGxB9SM'}
params={'metrics': 'ym:s:visits,ym:s:users,ym:s:pageviews,ym:s:bounceRate,ym:s:pageDepth,ym:s:avgVisitDurationSeconds',
        'dimensions': 'ym:s:date',
        'group': 'day',
        'date1': '2016-05-07',
        'date2': '2019-05-07',
        'limit': '100000',
        'filters': "ym:s:isNewUser=='Yes'",
        'sort': 'ym:s:date',
        'ids': counter,
        'accuracy': 'full',      
        'pretty': True}

print('Features: '+'Data,'+'Visits,'+'Viewers,'+'Views,'+'BR,'+'Depth'+'Time')

#get data#
r=requests.get('https://api-metrika.yandex.ru/stat/v1/data', params=params, headers=header)
parsed=json.loads(r.text)
##print (json.dumps(parsed, indent=4, sort_keys=True))
#get data#

#create dataframe/list#
column_names=['Data','Visits','Viewers','Views','BR','Depth','Time']
data_df=pd.DataFrame(columns=column_names);
data_list=[]
#/create dataframe/list#

#write to dataframe#
for day in parsed['data']:
    temp_dict={}
    temp_dict.update({'Data':day['dimensions'][0]['name'],'Visits':day['metrics'][0],'Viewers':day['metrics'][1],'Views':day['metrics'][2],'BR':day['metrics'][3],'Depth':day['metrics'][4],'Time':day['metrics'][5]})
    data_list.append(temp_dict);

data_df=pd.DataFrame(data_list,columns=column_names)    
absolute_path=os.path.dirname(os.path.abspath(__file__))

absolute_path=absolute_path+'/DATA/report.xlsx'
data_df.to_excel(absolute_path,index=False)
#/write to dataframe#  

#finishing alert#
print('\nFinished')
#/finishing alert#



