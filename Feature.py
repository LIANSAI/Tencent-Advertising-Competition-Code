# -*- coding: utf-8 -*-
"""
Created on Sat May 19 12:41:35 2018

@author: Allen
"""


import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os
import numpy as np
import gc
#from StringIO import StringIO
#from datalab.context import Context
#import datalab.storage as storage

#%gcs read --object gs://tencent666/adFeature.csv --variable adFeature
#%gcs read --object gs://tencent666/train.csv --variable train
#%gcs read --object gs://tencent666/test1.csv --variable test1
#%gcs read --object gs://tencent666/userFeature.csv --variable userFeature

#ad_feature = pd.read_csv(StringIO(adFeature))
#ad_feature.columns=['aid','advertsierid','campaginid','creativeid','creativesize','adcategoryid','productid','producttype']
#train = pd.read_csv(StringIO(train))
#prediict = pd.read_csv(StringIO(test1))
#user_feature = pd.read_csv(StringIO(userFeature))
   
# Memory Usage

types_dict_train = {'aid': 'int16',
             'uid': 'int32',
             'label': 'int8'
             }

train=pd.read_csv('file:///D:/数据/Kaggle/preliminary_contest_data(1)/preliminary_contest_data/train.csv',dtype=types_dict_train,nrows=10000)

types_dict_predict = {'aid':'int16',
                      'uid':'int32'
                      }

predict = pd.read_csv('file:///D:/数据/Kaggle/preliminary_contest_data(1)/preliminary_contest_data/test2.csv',dtype=types_dict_predict,nrows=20000)

types_dict_userfeature = {'LBS': 'float16',
 'age': 'int8',
 'appIdAction': 'O',
 'appIdInstall': 'O',
 'carrier': 'int8',
 'consumptionAbility': 'int8',
 'ct': 'O',
 'education': 'int8',
 'gender': 'int8',
 'house': 'float16',
 'interest1': 'O',
 'interest2': 'O',
 'interest3': 'O',
 'interest4': 'O',
 'interest5': 'O',
 'kw1': 'O',
 'kw2': 'O',
 'kw3': 'O',
 'marriageStatus': 'O',
 'os': 'O',
 'topic1': 'O',
 'topic2':'O',
 'topic3': 'O',
 'uid': 'int32'}

user_feature = pd.read_csv('file:///D:/数据/Kaggle/preliminary_contest_data(1)/preliminary_contest_data/userFeature.csv',dtype=types_dict_userfeature,nrows=200000)

types_dict_ad_feature={'adCategoryId': 'int16',
 'advertiserId': 'int32',
 'aid': 'int16',
 'campaignId': 'int32',
 'creativeId': 'int32',
 'creativeSize': 'int8',
 'productId': 'int16',
 'productType': 'int8'}

ad_feature = pd.read_csv('file:///D:/数据/Kaggle/preliminary_contest_data(1)/preliminary_contest_data/adFeature.csv',dtype=types_dict_ad_feature)
ad_feature.columns=['aid','advertsierid','campaginid','creativeid','creativesize','adcategoryid','productid','producttype']


train.label.replace(-1,0,inplace=True)
predict['label']=-1

##################User Feature Combination Only
#人群细分 #针对男性女性
user_feature['age_gender'] = user_feature.age.astype('str')+':'+user_feature.gender.astype('str')
user_feature['consump_gender'] = user_feature.consumptionAbility.astype('str')+':'+user_feature.gender.astype('str')
user_feature['edu_gender'] = user_feature.education.astype('str')+':'+user_feature.gender.astype('str')


#
# ct
def ctsort4(s):
    if '4' in s:
        return 1
    else:
        return 0
def ctsort3(s):
    if '3' in s:
        return 1
    else:
        return 0
def ctsort2(s):
    if '2' in s:
        return 1
    else:
        return 0
def ctsort1(s):
    if '1' in s:
        return 1
    else:
        return 0
user_feature['G4']= user_feature.ct.apply(ctsort4)
user_feature['G3']= user_feature.ct.apply(ctsort3)
user_feature['G2']= user_feature.ct.apply(ctsort2)
user_feature['WIFI']=user_feature.ct.apply(ctsort1)




#组合特征
predict['label']=-1
data = pd.concat([train,predict])
data= pd.merge(data,ad_feature,on='aid',how='left')
data= pd.merge(data,user_feature,on='uid',how='left')


data['age_gender'] = user_feature.age.astype('str')+':'+user_feature.gender.astype('str')

data['4G_creasize']=data.G4.astype('str')+':'+data.creativesize.astype('str')
data['3G_creasize']=data.G3.astype('str')+':'+data.creativesize.astype('str')
data['2G_creasize']=data.G2.astype('str')+':'+data.creativesize.astype('str')
data['wifi_creasize']=data.WIFI.astype('str')+':'+data.creativesize.astype('str')

del data,predict,train
gc.collet()



#############统计变量

####PoRDUCTtype
train_uafeature = pd.merge(train,ad_feature,on='aid',how='left')
train_uafeature = pd.merge(train_uafeature,user_feature,on='uid',how='left')

#种类
gender_type_feature = train_uafeature[['producttype']]
product_type = list(train_uafeature.producttype.unique())

m1 = train_uafeature[(train_uafeature.label==1)&(train_uafeature.gender==1)][['producttype']]
m1['prodtype_gender1']=1
m1 = m1.groupby('producttype').agg('sum').reset_index()

m2 = train_uafeature[(train_uafeature.label==1)&(train_uafeature.gender==2)][['producttype']]
m2['prodtype_gender2']=1
m2 = m2.groupby('producttype').agg('sum').reset_index()

m3= train_uafeature[(train_uafeature.gender==1)][['producttype']]
m3['prodtype_gender1_total']=1
m3 = m3.groupby('producttype').agg('sum').reset_index()

m4= train_uafeature[(train_uafeature.gender==2)][['producttype']]
m4['prodtype_gender2_total']=1
m4 = m4.groupby('producttype').agg('sum').reset_index()

for i in range(0,6):
    m5 = train_uafeature[(train_uafeature.label==1)&(train_uafeature.age==i)][['producttype']]
    m5['prodtype_age_seed'+str(i)]=1
    m5 = m5.groupby('producttype').agg('sum').reset_index()
    gender_type_feature =pd.merge(gender_type_feature,m5,on=['producttype'],how='left')

for i in range(0,6):
    m6 = train_uafeature[train_uafeature.age==i][['producttype']]
    m6['prodtype_age_total'+str(i)]=1
    m6 = m6.groupby('producttype').agg('sum').reset_index()
    gender_type_feature =pd.merge(gender_type_feature,m6,on=['producttype'],how='left')
    

gender_type_feature =pd.merge(gender_type_feature,m1,on=['producttype'],how='left')
gender_type_feature =pd.merge(gender_type_feature,m2,on=['producttype'],how='left')
gender_type_feature =pd.merge(gender_type_feature,m3,on=['producttype'],how='left')
gender_type_feature =pd.merge(gender_type_feature,m4,on=['producttype'],how='left')

for i in range(0,6):
    gender_type_feature['prodtype_ageconvertrate'+str(i)] = gender_type_feature['prodtype_age_seed'+str(i)].astype('float32')/gender_type_feature['prodtype_age_total'+str(i)]

gender_prodtype_total = gender_type_feature.prodtype_age_seed1+gender_type_feature.prodtype_age_seed2+gender_type_feature.prodtype_age_seed3+gender_type_feature.prodtype_age_seed4+gender_type_feature.prodtype_age_seed5
for i in range(1,6):
    gender_type_feature['prodtype_agerelative_rate'+str(i)] = gender_type_feature['prodtype_age_seed'+str(i)].astype('float32')/gender_prodtype_total


m7 = train_uafeature[(train_uafeature.label==1)&(train_uafeature.gender==1)][['producttype']]
m7['prodtype_gender1']=1
m7 = m7.groupby('producttype').agg('sum').reset_index()

m8 = train_uafeature[(train_uafeature.label==1)&(train_uafeature.gender==2)][['producttype']]
m8['prodtype_gender2']=1
m8 = m8.groupby('producttype').agg('sum').reset_index()

m8= train_uafeature[(train_uafeature.gender==1)][['producttype']]
m8['prodtype_gender1_total']=1
m8 = m8.groupby('producttype').agg('sum').reset_index()

m9= train_uafeature[(train_uafeature.gender==2)][['producttype']]
m9['prodtype_gender2_total']=1
m9 = m9.groupby('producttype').agg('sum').reset_index()


gender_type_feature['prodtype_consump_rate']= gender_type_feature.prodtype_gender1.astype('float32')/(gender_type_feature.prodtype_gender2+gender_type_feature.prodtype_gender1)
gender_type_feature['prodtype_1_consump'] = gender_type_feature.prodtype_gender1.astype('float32')/gender_type_feature.prodtype_gender1_total
gender_type_feature['prodtype_2_consump'] = gender_type_feature.prodtype_gender2.astype('float32')/gender_type_feature.prodtype_gender2_total
gender_type_feature = gender_type_feature.groupby('producttype').agg('mean').reset_index()
gender_type_feature=gender_type_feature.iloc[:,18:]
gender_type_feature['producttype'] =product_type

gender_type_feature.to_csv('producttype_feature.csv',index=None)



## os操作系统
user_feature.os.value_counts()
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
train_uafeature['os'] = train_uafeature.os.astype('str')

train_uafeature['os_labelenc'] = LabelEncoder().fit_transform(train_uafeature['os'])
train_uafeature.os.unique()
oslist = list(train_uafeature.os_labelenc.unique())
##################
os_feature = train_uafeature[['aid']].drop_duplicates()

for i in oslist:
    
    m1 = train_uafeature[(train_uafeature.label==1)&(train_uafeature.os_labelenc==i)][['aid']]
    m1['os_type_seed'+str(i)]=1
    m1 = m1.groupby('aid').agg('sum').reset_index()
    m2 = train_uafeature[train_uafeature.os_labelenc==i][['aid']]
    m2['os_type_total'+str(i)]=1
    m2 = m2.groupby('aid').agg('sum').reset_index()
    os_feature =pd.merge(os_feature,m1,on=['aid'],how='left')
    os_feature =pd.merge(os_feature,m2,on=['aid'],how='left')
    os_feature['os_type_convertrate'+str(i)] = os_feature['os_type_seed'+str(i)].astype('float32')/os_feature['os_type_total'+str(i)]
    os_feature.drop(['os_type_total'+str(i),'os_type_seed'+str(i)],inplace=True,axis=1)

os_feature.fillna(0,inplace=True)      
os_feature.to_csv('os_feature.csv',index=None)

## 婚姻
train_uafeature['marriageStatus'] = train_uafeature.marriageStatus.astype('str')
train_uafeature['marriageStatus_encoder'] = LabelEncoder().fit_transform(train_uafeature['marriageStatus'])
marrylist = list(train_uafeature.marriageStatus_encoder.unique())
marry_feature = train_uafeature[['aid']].drop_duplicates()

for i in marrylist:
    
    m1 = train_uafeature[(train_uafeature.label==1)&(train_uafeature.marriageStatus_encoder==i)][['aid']]
    m1['marry_type_seed'+str(i)]=1
    m1 = m1.groupby('aid').agg('sum').reset_index()
    m2 = train_uafeature[train_uafeature.marriageStatus_encoder==i][['aid']]
    m2['marry_type_total'+str(i)]=1
    m2 = m2.groupby('aid').agg('sum').reset_index()
    marry_feature =pd.merge(marry_feature,m1,on=['aid'],how='left')
    marry_feature =pd.merge(marry_feature,m2,on=['aid'],how='left')
    marry_feature['marry_type_convertrate'+str(i)] = marry_feature['marry_type_seed'+str(i)].astype('float32')/marry_feature['marry_type_total'+str(i)]
    marry_feature.drop(['marry_type_total'+str(i),'marry_type_seed'+str(i)],inplace=True,axis=1)

marry_feature.fillna(0,inplace=True)      
marry_feature.to_csv('marry_feature.csv',index=None)

# 地理位置
train_uafeature['LBS'] = train_uafeature.LBS.astype('str')
train_uafeature['LBS_encoder'] = LabelEncoder().fit_transform(train_uafeature['LBS'])
len(train_uafeature.LBS.unique())

lbslist = list(train_uafeature.LBS_encoder.unique())
lbs_feature = train_uafeature[['aid']].drop_duplicates()

for i in lbslist:
    
    m1 = train_uafeature[(train_uafeature.label==1)&(train_uafeature.LBS_encoder==i)][['aid']]
    m1['lbs_seed'+str(i)]=1
    m1 = m1.groupby('aid').agg('sum').reset_index()
    m2 = train_uafeature[train_uafeature.LBS_encoder==i][['aid']]
    m2['lbs_total'+str(i)]=1
    m2 = m2.groupby('aid').agg('sum').reset_index()
    lbs_feature =pd.merge(lbs_feature,m1,on=['aid'],how='left')
    lbs_feature =pd.merge(lbs_feature,m2,on=['aid'],how='left')
    lbs_feature['lbs_convertrate'+str(i)] = lbs_feature['lbs_seed'+str(i)].astype('float')/lbs_feature['lbs_total'+str(i)]
    lbs_feature.drop(['lbs_seed'+str(i),'lbs_total'+str(i)],inplace=True,axis=1)

lbs_feature.fillna(0,inplace=True)      
lbs_feature.to_csv('lbs_feature.csv',index=None)

#移动运营商
train_uafeature['carrier'] = train_uafeature.carrier.astype('str')
train_uafeature['carrier_encode'] = LabelEncoder().fit_transform(train_uafeature['carrier'])
len(train_uafeature.carrier.unique())

carrierlist = list(train_uafeature.carrier_encode.unique())
carrier_feature = train_uafeature[['aid']].drop_duplicates()

for i in carrierlist:
    
    m1 = train_uafeature[(train_uafeature.label==1)&(train_uafeature.carrier_encode==i)][['aid']]
    m1['carrier_seed'+str(i)]=1
    m1 = m1.groupby('aid').agg('sum').reset_index()
    m2 = train_uafeature[train_uafeature.carrier_encode==i][['aid']]
    m2['carrier_total'+str(i)]=1
    m2 = m2.groupby('aid').agg('sum').reset_index()
    carrier_feature =pd.merge(carrier_feature,m1,on=['aid'],how='left')
    carrier_feature =pd.merge(carrier_feature,m2,on=['aid'],how='left')
    carrier_feature['carrier_convertrate'+str(i)] = carrier_feature['carrier_seed'+str(i)].astype('float')/carrier_feature['carrier_total'+str(i)]
    carrier_feature.drop(['carrier_seed'+str(i),'carrier_total'+str(i)],inplace=True,axis=1)

carrier_feature.fillna(0,inplace=True)      
carrier_feature.to_csv('carrier_feature.csv',index=None)


#############人群细分
###消费
train_uafeature['consump_gender'] = train_uafeature.age_gender.astype('str')
train_uafeature['consump_gender'] = LabelEncoder().fit_transform(train_uafeature['consump_gender'])
len(train_uafeature.consump_gender.unique())

consump_genderlist = list(train_uafeature.consump_gender.unique())
consump_gender_feature = train_uafeature[['consump_gender']].drop_duplicates()

for i in consump_genderlist:
    
    m1 = train_uafeature[(train_uafeature.label==1)&(train_uafeature.consump_gender==i)][['aid']]
    m1['consump_gender_seed'+str(i)]=1
    m1 = m1.groupby('aid').agg('sum').reset_index()
    m2 = train_uafeature[train_uafeature.consump_gender==i][['aid']]
    m2['consump_gender_total'+str(i)]=1
    m2 = m2.groupby('aid').agg('sum').reset_index()
    consump_gender_feature =pd.merge(consump_gender_feature,m1,on=['aid'],how='left')
    consump_gender_feature =pd.merge(consump_gender_feature,m2,on=['aid'],how='left')
    consump_gender_feature['consump_gender_convertrate'+str(i)] = consump_gender_feature['consump_gender_seed'+str(i)].astype('float')/consump_gender_feature['consump_gender_total'+str(i)]
    consump_gender_feature.drop(['consump_gender_seed'+str(i),'consump_gender_total'+str(i)],inplace=True,axis=1)

consump_gender_feature.fillna(0,inplace=True)      
consump_gender_feature.to_csv('consump_gender_feature.csv',index=None)



####age男性女性
train_uafeature['age_gender'] = train_uafeature.age_gender.astype('str')
train_uafeature['age_gender_encode'] = LabelEncoder().fit_transform(train_uafeature['age_gender'])
len(train_uafeature.age_gender_encode.unique())

agegenderrlist = list(train_uafeature.age_gender_encode.unique())
age_gender_feature = train_uafeature[['aid']].drop_duplicates()

for i in agegenderrlist:
    
    m1 = train_uafeature[(train_uafeature.label==1)&(train_uafeature.age_gender_encode==i)][['aid']]
    m1['age_gender_seed'+str(i)]=1
    m1 = m1.groupby('aid').agg('sum').reset_index()
    m2 = train_uafeature[train_uafeature.age_gender_encode==i][['aid']]
    m2['age_gender_total'+str(i)]=1
    m2 = m2.groupby('aid').agg('sum').reset_index()
    age_gender_feature =pd.merge(age_gender_feature,m1,on=['aid'],how='left')
    age_gender_feature =pd.merge(age_gender_feature,m2,on=['aid'],how='left')
    age_gender_feature['age_gender_convertrate'+str(i)] = age_gender_feature['age_gender_seed'+str(i)].astype('float')/age_gender_feature['age_gender_total'+str(i)]
    age_gender_feature.drop(['age_gender_seed'+str(i),'age_gender_total'+str(i)],inplace=True,axis=1)

age_gender_feature.fillna(0,inplace=True)      
age_gender_feature.to_csv('age_gender_feature.csv',index=None)





############广告对不同地理位置的转化率##############
lbs = list(train_uafeature.LBS.unique())
ad = list(train_uafeature.aid.unique())
#lbs_aid = train_uafeature['uid']
lbs_feature = train_uafeature[['aid','LBS']]
for a in ad:
    a=1991
    m1 = train_uafeature[(train_uafeature.aid==a)&(train_uafeature.label==1)][['LBS']]
    m1['lbs_seed_total'+str(a)] = 1
    m1 = m1.groupby('LBS').agg('sum').reset_index()
    m1['aid'] = a
    m2 = train_uafeature[train_uafeature.aid==a][['LBS']]
    m2['lbs_total'+str(a)] = 1
    m2 = m2.groupby('LBS').agg('sum').reset_index()
    m2['aid'] = a
    lbs_feature = pd.merge(lbs_feature,m1,on=['aid','LBS'],how='left')
    lbs_feature = pd.merge(lbs_feature,m2,on=['aid','LBS'],how='left')
    lbs_feature['lbs_covert_rate'+str(a)] = lbs_feature['lbs_seed_total'+str(a)].astype('float')/lbs_feature['lbs_total'+str(a)]
    lbs_feature.drop(['lbs_total'+str(a),'lbs_seed_total'+str(a)],inplace=True,axis=1)



#User组合
#特征组合
def combinefeature(x,y):
    x=str(x)
    y=str(y)
    return x+';'+y

user_feature['age_gender'] = user_feature.age.astype('str')+':'+user_feature.gender.astype('str')
user_feature['consump_gender'] = user_feature.consumptionAbility.astype('str')+':'+user_feature.gender.astype('str')
