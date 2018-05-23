# -*- coding: utf-8 -*-
"""
Created on Wed May 23 12:44:53 2018

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
from io import BytesIO as StringIO
from datalab.context import Context
import datalab.storage as storage


types_dict_ad_feature={'adCategoryId': 'int16',
 'advertiserId': 'int32',
 'aid': 'int16',
 'campaignId': 'int32',
 'creativeId': 'int32',
 'creativeSize': 'int8',
 'productId': 'int16',
 'productType': 'int8'}

ad_feature = pd.read_csv(StringIO(adFeature),dtype=types_dict_ad_feature)
ad_feature.columns=['aid','advertsierid','campaginid','creativeid','creativesize','adcategoryid','productid','producttype']

types_dict_train = {'aid': 'int16',
             'uid': 'int32',
             'label': 'int8'
             }

train = pd.read_csv(StringIO(train),dtype=types_dict_train)
mem = train.memory_usage(index=True).sum()
print(mem/ 1024**2," MB")

types_dict_predict = {'aid':'int16',
                      'uid':'int32'
                      }

predict = pd.read_csv(StringIO(test2),dtype=types_dict_predict)

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
user_feature = pd.read_csv(StringIO(userFeature),dtype=types_dict_userfeature)

age_feature = pd.read_csv('age_feature.csv')
listf = list(age_feature.columns)
listf.remove('aid')
listf1 = listf[:6]
listf2 = listf[6:]
for i in listf1:
  age_feature[i]=age_feature[i].astype('float16')
for i in listf2:
  age_feature[i]=age_feature[i].astype('int8')


consumption_feature = pd.read_csv('consumption_feature.csv')
listf = list(consumption_feature.columns)
listf.remove('aid')
for i in listf:
  consumption_feature[i]=consumption_feature[i].astype('float16')
  
edu_feature = pd.read_csv('edu_feature.csv')
edu_feature.drop('aid.1',inplace=True,axis=1)
listf = list(edu_feature.columns)
listf.remove('aid')
listf1=listf[:4]
listf2=listf[4:]
for i in listf1:
  edu_feature[i]=edu_feature[i].astype('int8')
for i in listf2:
  edu_feature[i]=edu_feature[i].astype('float16')
  
gender_feature =pd.read_csv('gender_feature.csv') 
listf = list(gender_feature.columns)
listf.remove('aid')
for i in listf:
  gender_feature[i]=gender_feature[i].astype('float16')
  
house_feature = pd.read_csv('house_feature.csv')


age_gender_feature = pd.read_csv('age_gender_feature.csv')
listf = list(age_gender_feature.columns)
listf.remove('aid')
for i in listf:
  age_gender_feature[i]=age_gender_feature[i].astype('float16')

producttype_feature = pd.read_csv('producttype_feature.csv')

os_feature = pd.read_csv('os_feature.csv')
listf = list(os_feature.columns)
listf.remove('aid')
for i in listf:
  os_feature[i]=os_feature[i].astype('float16')

marry_feature = pd.read_csv('marry_feature.csv')
listf = list(marry_feature.columns)
listf.remove('aid')
for i in listf:
  marry_feature[i]=marry_feature[i].astype('float16')

carrier_feature = pd.read_csv('carrier_feature.csv')
listf = list(carrier_feature.columns)
listf.remove('aid')
for i in listf:
  carrier_feature[i]=carrier_feature[i].astype('float16')

consump_gender_feature = pd.read_csv('consump_gender_feature.csv')
listf = list(consump_gender_feature.columns)
listf.remove('aid')
for i in listf:
  consump_gender_feature[i]=consump_gender_feature[i].astype('float16')


lbs_feature = pd.read_csv('lbs_feature.csv')
listf = list(lbs_feature.columns)
listf.remove('aid')
for i in listf:
  lbs_feature[i]=lbs_feature[i].astype('float16')


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


train.label.replace(-1,0,inplace=True)
predict['label']=-1
predict['label']=predict['label'].astype('int8')

def main(train1,valid,predict):
    data1 = pd.concat([train1,valid])
    data1 = pd.concat([data1,predict])
    data1 = pd.merge(data1,ad_feature,on='aid',how='left')
    data1 = pd.merge(data1,user_feature,on='uid',how='left')
    data1['4G_creasize']=data1.G4.astype('str')+':'+data1.creativesize.astype('str')
    data1['3G_creasize']=data1.G3.astype('str')+':'+data1.creativesize.astype('str')
    data1['2G_creasize']=data1.G2.astype('str')+':'+data1.creativesize.astype('str')
    data1['wifi_creasize']=data1.WIFI.astype('str')+':'+data1.creativesize.astype('str')
    
    data1=data1.fillna('-1')  
    
    one_hot_feature=['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','marriageStatus','advertsierid','campaginid', 'creativeid',
       'adcategoryid', 'productid','4G_creasize','3G_creasize','2G_creasize','wifi_creasize']
    
    vector_feature=['appIdAction','appIdInstall','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3']
   
    for feature in one_hot_feature:
        try:
            data1[feature] = LabelEncoder().fit_transform(data1[feature].apply(int))
        except:
            data1[feature] = LabelEncoder().fit_transform(data1[feature])
    
    data1['LBS']=data1['LBS'].astype('int16')
    data1['adcategoryid']=data1['adcategoryid'].astype('int8')
    data1['advertsierid']=data1['advertsierid'].astype('int32')
    data1['age']=data1['age'].astype('int8')
    data1['campaginid']=data1['campaginid'].astype('int32')
    data1['carrier']=data1['carrier'].astype('int8')
    data1['consumptionAbility']=data1['consumptionAbility'].astype('int8')
    data1['creativeid']=data1['creativeid'].astype('int16')
    data1['creativesize']=data1['creativesize'].astype('int8')
    data1['ct']=data1['ct'].astype('int8')
    data1['education']=data1['education'].astype('int8')
    data1['gender']=data1['gender'].astype('int8')
    data1['house']=data1['house'].astype('int8')
    data1['marriageStatus']=data1['marriageStatus'].astype('int8')
    data1['os']=data1['os'].astype('int8')
    data1['productid']=data1['productid'].astype('int8')
    data1['producttype']=data1['producttype'].astype('int8')
    
    train1=data1[data1.label!=-1]
    train_train = train1[train1.label!=-2]
    train_y=train_train.pop('label')

    train_valid = train1[train1.label==-2]

    test=data1[data1.label==-1]
    
    
    res_test=test[['aid','uid']]
    res_valid=train_valid[['aid','uid']]

    test=test.drop('label',axis=1)
    train_valid = train_valid.drop('label',axis=1)
    enc = OneHotEncoder()
    train_x=train_train[['aid','uid','creativesize','producttype']]
    valid_x=train_valid[['aid','uid','creativesize','producttype']]
    test_x=test[['aid','uid','creativesize','producttype']]
    
    print('Split done')
    print(train_x.shape,valid_x.shape,test_x.shape)

    train_x = pd.merge(train_x,age_feature,on='aid',how='left')
    train_x = pd.merge(train_x,consumption_feature,on='aid',how='left')
    train_x = pd.merge(train_x,edu_feature,on='aid',how='left')
    train_x = pd.merge(train_x,gender_feature,on='aid',how='left')
    train_x = pd.merge(train_x,house_feature,on='aid',how='left')
    train_x = pd.merge(train_x,age_gender_feature,on='aid',how='left')
    train_x = pd.merge(train_x,producttype_feature,on='producttype',how='left')
    train_x = pd.merge(train_x,os_feature,on='aid',how='left')
    train_x = pd.merge(train_x,marry_feature,on='aid',how='left')
    train_x = pd.merge(train_x,carrier_feature,on='aid',how='left')
    #train_x = pd.merge(train_x,lbs_feature,on='aid',how='left')
    #train_x = pd.merge(train_x,consump_gender_feature,on='aid',how='left')
    
    valid_x = pd.merge(valid_x,age_feature,on='aid',how='left')
    valid_x = pd.merge(valid_x,consumption_feature,on='aid',how='left')
    valid_x = pd.merge(valid_x,edu_feature,on='aid',how='left')
    valid_x = pd.merge(valid_x,gender_feature,on='aid',how='left')
    valid_x = pd.merge(valid_x,house_feature,on='aid',how='left')
    valid_x = pd.merge(valid_x,age_gender_feature,on='aid',how='left')
    valid_x = pd.merge(valid_x,producttype_feature,on='producttype',how='left')
    valid_x = pd.merge(valid_x,os_feature,on='aid',how='left')
    valid_x = pd.merge(valid_x,marry_feature,on='aid',how='left')
    valid_x = pd.merge(valid_x,carrier_feature,on='aid',how='left')
    #valid_x = pd.merge(valid_x,lbs_feature,on='aid',how='left')
    #valid_x = pd.merge(valid_x,consump_gender_feature,on='aid',how='left')
    
    test_x = pd.merge(test_x,age_feature,on='aid',how='left')
    test_x = pd.merge(test_x,consumption_feature,on='aid',how='left')
    test_x = pd.merge(test_x,edu_feature,on='aid',how='left')
    test_x = pd.merge(test_x,gender_feature,on='aid',how='left')
    test_x = pd.merge(test_x,house_feature,on='aid',how='left')
    test_x = pd.merge(test_x,age_gender_feature,on='aid',how='left')
    test_x = pd.merge(test_x,producttype_feature,on='producttype',how='left')
    test_x = pd.merge(test_x,os_feature,on='aid',how='left')
    test_x = pd.merge(test_x,marry_feature,on='aid',how='left')
    test_x = pd.merge(test_x,carrier_feature,on='aid',how='left')
    #test_x = pd.merge(test_x,consump_gender_feature,on='aid',how='left')
    #test_x = pd.merge(test_x,lbs_feature,on='aid',how='left')
    
    train_x.drop(['aid','uid','producttype'],inplace=True,axis=1)
    test_x.drop(['aid','uid','producttype'],inplace=True,axis=1)
    valid_x.drop(['aid','uid','producttype'],inplace=True,axis=1)
    
    print('Merge done')

    
    print(train_x.shape,valid_x.shape,test_x.shape)
    
    
    for feature in one_hot_feature:
        enc.fit(data1[feature].values.reshape(-1, 1))
        train_a=enc.transform(train_train[feature].values.reshape(-1, 1))
        valid_a=enc.transform(train_valid[feature].values.reshape(-1, 1))
        test_a = enc.transform(test[feature].values.reshape(-1, 1))
        train_x= sparse.hstack((train_x, train_a))
        valid_x = sparse.hstack((valid_x, valid_a))
        test_x = sparse.hstack((test_x, test_a))
  
    print('one-hot prepared !')
    
    print(train_x.shape,valid_x.shape,test_x.shape)

    cv=CountVectorizer(max_features=700)
    for feature in vector_feature:
        cv.fit(data1[feature])
        train_a = cv.transform(train_train[feature])
        valid_a = cv.transform(train_valid[feature])
        test_a = cv.transform(test[feature])
       
        train_x = sparse.hstack((train_x, train_a))
        valid_x = sparse.hstack((valid_x, valid_a))
        test_x = sparse.hstack((test_x, test_a))
        print(feature,'done')
    print('cv prepared !')
    

    print(train_x.shape,valid_x.shape,test_x.shape)
    
    return train_x,train_y,valid_x,test_x,res_test,res_valid

def LGB_predict(train_x,train_y,valid_x,test_x,res_test,res_valid):
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=2000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,num_threads=8,feature_fraction=0.8,max_bin=240,
        learning_rate=0.040, min_child_weight=50, random_state=2018, n_jobs=100
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc',early_stopping_rounds=100)
    res_test['score'] = clf.predict_proba(test_x)[:,1]
    res_valid['score'] = clf.predict_proba(valid_x)[:,1]

    res_valid['score'] = res_valid['score'].apply(lambda x: float('%.6f' % x))
    res_test['score'] = res_test['score'].apply(lambda x: float('%.6f' % x))
    
  
    
    return res_valid,res_test,clf


train_y_total = train['label']
train_y_part1 = train_y_total.iloc[:2932938]
train_part1 = train.iloc[:2932938,:]
valid = train_part1.iloc[1466469:]
valid_y = valid.pop('label')
valid['label'] = -2
train1 = train_part1.iloc[:1466469]

train_x,train_y,valid_x,test_x,res_test,res_valid=main(train1,valid,predict)

res_valid,res_test,model = LGB_predict_low_depth(train_x,train_y,valid_x,test_x,res_test,res_valid)
