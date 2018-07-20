# -*- coding: utf-8 -*-
"""
Created on Sun May 13 11:23:08 2018

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

gender = pd.read_csv('gender_feature.csv')
a=gender.head(100)

ad_feature=pd.read_csv('file:///D:/数据/Kaggle/preliminary_contest_data(1)/preliminary_contest_data/adFeature.csv')
ad_feature.columns=['aid','advertsierid','campaginid','creativeid','creativesize','adcategoryid','productid','producttype']
if os.path.exists('file:///D:/数据/Kaggle/preliminary_contest_data(1)/preliminary_contest_data/userFeature.csv'):
    user_feature=pd.read_csv('file:///D:/数据/Kaggle/preliminary_contest_data(1)/preliminary_contest_data/userFeature.csv',nrows=1000000)
else:
    userFeature_data = []
    with open('userFeature.data', 'r') as f:
        cnt = 0
        for i, line in enumerate(f):
            line = line.strip().split('|')
            userFeature_dict = {}
            for each in line:
                each_list = each.split(' ')
                userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
            userFeature_data.append(userFeature_dict)
            if i % 100000 == 0:
                print(i)
            if i % 1000000 == 0:
                user_feature = pd.DataFrame(userFeature_data)
                user_feature.to_csv('userFeature_' + str(cnt) + '.csv', index=False)
                cnt += 1
                del userFeature_data, user_feature
                userFeature_data = []
        user_feature = pd.DataFrame(userFeature_data)
        user_feature.to_csv('userFeature_' + str(cnt) + '.csv', index=False)
        del userFeature_data, user_feature
        user_feature = pd.concat([pd.read_csv('userFeature_' + str(i) + '.csv') for i in range(cnt + 1)]).reset_index(drop=True)
        user_feature.to_csv('userFeature.csv', index=False)
   
head=user_feature.head(100)
train=pd.read_csv('file:///D:/数据/Kaggle/preliminary_contest_data(1)/preliminary_contest_data/train.csv')
predict=pd.read_csv('file:///D:/数据/Kaggle/preliminary_contest_data(1)/preliminary_contest_data/test1.csv')        

train.label.replace(-1,0,inplace=True)

train1=train[:1000000]
user_feature = user_feature[:1000000]
train_adfeature = pd.merge(train1,ad_feature,on='aid',how='left')
train_adfeature = pd.merge(train_adfeature,user_feature,on='uid',how='left')



###### Ad Feature########
#每一个广告有多少用户
t1=train[['aid']]
t1['total customer_each_aid'] =1
t1 = t1.groupby('aid').agg('sum').reset_index()
#每一个广告有多少种子用户
t2=train[['aid','label']]
t2 = t2.groupby('aid').agg('sum').reset_index()
#种子客户率

#商品ID
train_adfeature.productid.value_counts()
v=train_adfeature.head(100)
t1 = train_adfeature[train_adfeature.label==1][['productid']]
t1['counts']
t1= t1.groupby('productid')
v=train_adfeature.head()
###### 用户广告feature（寻找广告的用户群体）########
########

train_uafeature = pd.merge(train1,user_feature,on='uid',how='left')


######################################特征工程###################################
##################### User Feature ####################
#############对男女群体的转化率########
#一个广告种子客户中男女性占比，男性转化率，女性转化率
gender_feature = train_uafeature[['aid']]
#男性1号人数
m1 = train_uafeature[(train_uafeature.gender==1)&(train_uafeature.label==1)][['aid']]
m1['male_seed_num']=1
m1 = m1.groupby('aid').agg('sum').reset_index()
#女性1号人数
m2 = train_uafeature[(train_uafeature.gender==2)&(train_uafeature.label==1)][['aid']]
m2['female_seed_num']=1
m2 = m2.groupby('aid').agg('sum').reset_index()
#男性1号总人数
m3 = train_uafeature[train_uafeature.gender==1][['aid']]
m3['male_num']=1
m3 = m3.groupby('aid').agg('sum').reset_index()
#女性2号总人数
m4 = train_uafeature[train_uafeature.gender==2][['aid']]
m4['female_num']=1
m4 = m4.groupby('aid').agg('sum').reset_index()

gender_feature =pd.merge(gender_feature,m1,on=['aid'],how='left')
gender_feature =pd.merge(gender_feature,m2,on=['aid'],how='left')
gender_feature =pd.merge(gender_feature,m3,on=['aid'],how='left')
gender_feature =pd.merge(gender_feature,m4,on=['aid'],how='left')
gender_feature['total_seed_num'] = gender_feature.male_seed_num+gender_feature.female_seed_num
gender_feature['gender_rate']=gender_feature.male_seed_num.astype('float')/gender_feature.total_seed_num
gender_feature['male_convert_rate']=gender_feature.male_seed_num.astype('float')/gender_feature.male_num
gender_feature['female_convert_rate']=gender_feature.female_seed_num.astype('float')/gender_feature.female_num
gender_feature=gender_feature[['aid','gender_rate','male_convert_rate','female_convert_rate']]
gender_feature.to_csv('gender_feature.csv',index=None)


###不同年龄群体转化率#########
age_feature = train_uafeature[['aid']]
#年龄5seed人数
m1 = train_uafeature[(train_uafeature.age==5)&(train_uafeature.label==1)][['aid']]
m1['age5_seed']=1
m1 = m1.groupby('aid').agg('sum').reset_index()
#年龄5总人数
m2 = train_uafeature[train_uafeature.age==5][['aid']]
m2['age5_total']=1
m2 = m2.groupby('aid').agg('sum').reset_index()

#年龄4seed人数
m3 = train_uafeature[(train_uafeature.age==4)&(train_uafeature.label==1)][['aid']]
m3['age4_seed']=1
m3 = m3.groupby('aid').agg('sum').reset_index()
#年龄4总人数
m4 = train_uafeature[train_uafeature.age==4][['aid']]
m4['age4_total']=1
m4 = m4.groupby('aid').agg('sum').reset_index()

#年龄3seed人数
m5 = train_uafeature[(train_uafeature.age==3)&(train_uafeature.label==1)][['aid']]
m5['age3_seed']=1
m5 = m5.groupby('aid').agg('sum').reset_index()
#年龄3总人数
m6 = train_uafeature[train_uafeature.age==3][['aid']]
m6['age3_total']=1
m6 = m6.groupby('aid').agg('sum').reset_index()

#年龄2seed人数
m7 = train_uafeature[(train_uafeature.age==2)&(train_uafeature.label==1)][['aid']]
m7['age2_seed']=1
m7 = m7.groupby('aid').agg('sum').reset_index()
#年龄2总人数
m8 = train_uafeature[train_uafeature.age==2][['aid']]
m8['age2_total']=1
m8 = m8.groupby('aid').agg('sum').reset_index()

#年龄1seed人数
m9 = train_uafeature[(train_uafeature.age==1)&(train_uafeature.label==1)][['aid']]
m9['age1_seed']=1
m9 = m9.groupby('aid').agg('sum').reset_index()
#年龄1总人数
m10 = train_uafeature[train_uafeature.age==1][['aid']]
m10['age1_total']=1
m10 = m10.groupby('aid').agg('sum').reset_index()

#年龄0seed人数
m11 = train_uafeature[(train_uafeature.age==0)&(train_uafeature.label==1)][['aid']]
m11['age0_seed']=1
m11 = m11.groupby('aid').agg('sum').reset_index()
#年龄0总人数
m12 = train_uafeature[train_uafeature.age==0][['aid']]
m12['age0_total']=1
m12 = m12.groupby('aid').agg('sum').reset_index()
#平均年龄层
m13 = train_uafeature[train_uafeature.label==1][['aid','age']]
m14 = m13.groupby('aid').agg('mean').reset_index()
m14.rename(columns={'age':'age_mean'},inplace=True)
#最大年龄层
m15 = m13.groupby('aid').agg('max').reset_index()
m15.rename(columns={'age':'age_max'},inplace=True)
#最小年龄层
m16 = m13.groupby('aid').agg('min').reset_index()
m16.rename(columns={'age':'age_min'},inplace=True)
#中位数
m17 = m13.groupby('aid').agg('median').reset_index()
m17.rename(columns={'age':'age_median'},inplace=True)

for i in [m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m14,m15,m16,m17]:
    age_feature = pd.merge(age_feature,i,on='aid',how='left')

age_feature['age5_convert_rate']=age_feature.age5_seed.astype('float')/age_feature.age5_total
age_feature['age4_convert_rate']=age_feature.age4_seed.astype('float')/age_feature.age4_total
age_feature['age3_convert_rate']=age_feature.age3_seed.astype('float')/age_feature.age3_total
age_feature['age2_convert_rate']=age_feature.age2_seed.astype('float')/age_feature.age2_total
age_feature['age1_convert_rate']=age_feature.age1_seed.astype('float')/age_feature.age1_total
age_feature['age0_convert_rate']=age_feature.age0_seed.astype('float')/age_feature.age0_total
  
age_feature=age_feature[['aid','age5_convert_rate','age4_convert_rate','age3_convert_rate','age2_convert_rate','age1_convert_rate','age0_convert_rate','age_mean','age_max','age_min','age_median']]
gender_feature.to_csv('gender_feature.csv',index=None)

del m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16,m17
gc.collect()

#######3######学历#############
edu_feature = train_uafeature[['aid']]

#学历0
m0 = train_uafeature[(train_uafeature.education==0)&(train_uafeature.label==1)][['aid']]
m0['edu_seed_count_0']=1
m0 = m0.groupby('aid').agg('sum').reset_index()
#学历1
m1 = train_uafeature[(train_uafeature.education==1)&(train_uafeature.label==1)][['aid']]
m1['edu_seed_count_1']=1
m1 = m1.groupby('aid').agg('sum').reset_index()

#学历2
m2 = train_uafeature[(train_uafeature.education==2)&(train_uafeature.label==1)][['aid']]
m2['edu_seed_count_2']=1
m2 = m2.groupby('aid').agg('sum').reset_index()
#学历3
m3 = train_uafeature[(train_uafeature.education==3)&(train_uafeature.label==1)][['aid']]
m3['edu_seed_count_3']=1
m3 = m3.groupby('aid').agg('sum').reset_index()
#学历4
m4 = train_uafeature[(train_uafeature.education==4)&(train_uafeature.label==1)][['aid']]
m4['edu_seed_count_4']=1
m4 = m4.groupby('aid').agg('sum').reset_index()
#学历5
m5 = train_uafeature[(train_uafeature.education==5)&(train_uafeature.label==1)][['aid']]
m5['edu_seed_count_5']=1
m5 = m5.groupby('aid').agg('sum').reset_index()
#学历6
m6 = train_uafeature[(train_uafeature.education==6)&(train_uafeature.label==1)][['aid']]
m6['edu_seed_count_6']=1
m6 = m6.groupby('aid').agg('sum').reset_index()
#学历7
m7= train_uafeature[(train_uafeature.education==7)&(train_uafeature.label==1)][['aid']]
m7['edu_seed_count_7']=1
m7 = m7.groupby('aid').agg('sum').reset_index()

#学历0总人数
m8 = train_uafeature[train_uafeature.education==0][['aid']]
m8['edu_total_count_0']=1
m8 = m8.groupby('aid').agg('sum').reset_index()
#1总人数
m9 = train_uafeature[train_uafeature.education==1][['aid']]
m9['edu_total_count_1']=1
m9 = m9.groupby('aid').agg('sum').reset_index()

m10 = train_uafeature[train_uafeature.education==2][['aid']]
m10['edu_total_count_2']=1
m10 = m10.groupby('aid').agg('sum').reset_index()

m11 = train_uafeature[train_uafeature.education==3][['aid']]
m11['edu_total_count_3']=1
m11 = m11.groupby('aid').agg('sum').reset_index()

m12 = train_uafeature[train_uafeature.education==4][['aid']]
m12['edu_total_count_4']=1
m12 = m12.groupby('aid').agg('sum').reset_index()

m13= train_uafeature[train_uafeature.education==5][['aid']]
m13['edu_total_count_5']=1
m13 = m13.groupby('aid').agg('sum').reset_index()

m14 = train_uafeature[train_uafeature.education==6][['aid']]
m14['edu_total_count_6']=1
m14 = m14.groupby('aid').agg('sum').reset_index()

m15= train_uafeature[train_uafeature.education==7][['aid']]
m15['edu_total_count_7']=1
m15 = m15.groupby('aid').agg('sum').reset_index()

#平均教育层
m16 = train_uafeature[train_uafeature.label==1][['aid','education']]
m17 = m16.groupby('aid').agg('mean').reset_index()
m17.rename(columns={'education':'education_mean'},inplace=True)
#最大教育层
m18 = m16.groupby('aid').agg('max').reset_index()
m18.rename(columns={'education':'education_max'},inplace=True)
#最小教育层
m19 = m16.groupby('aid').agg('min').reset_index()
m19.rename(columns={'education':'education_min'},inplace=True)
#中位数
m20 = m16.groupby('aid').agg('median').reset_index()
m20.rename(columns={'education':'education_median'},inplace=True)

for i in [m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m17,m18,m19,m20]:
    edu_feature = pd.merge(edu_feature,i,on='aid',how='left')

edu_feature['edu_0_convert_rate']=edu_feature.edu_seed_count_0.astype('float')/edu_feature.edu_total_count_0
edu_feature['edu_1_convert_rate']=edu_feature.edu_seed_count_1.astype('float')/edu_feature.edu_total_count_1
edu_feature['edu_2_convert_rate']=edu_feature.edu_seed_count_2.astype('float')/edu_feature.edu_total_count_2
edu_feature['edu_3_convert_rate']=edu_feature.edu_seed_count_3.astype('float')/edu_feature.edu_total_count_3
edu_feature['edu_4_convert_rate']=edu_feature.edu_seed_count_4.astype('float')/edu_feature.edu_total_count_4
edu_feature['edu_5_convert_rate']=edu_feature.edu_seed_count_5.astype('float')/edu_feature.edu_total_count_5
edu_feature['edu_6_convert_rate']=edu_feature.edu_seed_count_6.astype('float')/edu_feature.edu_total_count_6
edu_feature['edu_7_convert_rate']=edu_feature.edu_seed_count_7.astype('float')/edu_feature.edu_total_count_7

edu_feature=edu_feature[['aid','education_mean','education_max','education_min','education_median','aid','edu_0_convert_rate','edu_1_convert_rate','edu_2_convert_rate','edu_3_convert_rate','edu_4_convert_rate','edu_5_convert_rate','edu_6_convert_rate','edu_7_convert_rate']]
edu_feature.to_csv('edu_feature.csv',index=None)

b = edu_feature.head(100)

del m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16,m17,m18,m19,m20
gc.collect()

#################收入高低##################
#user_feature.consumptionAbility.value_counts()
#一个广告种子客户中收入占比，转化率
consumption_feature = train_uafeature[['aid']]
#消费0seed人数
m1 = train_uafeature[(train_uafeature.consumptionAbility==0)&(train_uafeature.label==1)][['aid']]
m1['consump_0_seed']=1
m1 = m1.groupby('aid').agg('sum').reset_index()
#消费1seed人数
m2 = train_uafeature[(train_uafeature.consumptionAbility==1)&(train_uafeature.label==1)][['aid']]
m2['consump_1_seed']=1
m2 = m2.groupby('aid').agg('sum').reset_index()
#消费2seed人数
m3 = train_uafeature[(train_uafeature.consumptionAbility==2)&(train_uafeature.label==1)][['aid']]
m3['consump_2_seed']=1
m3 = m3.groupby('aid').agg('sum').reset_index()

##消费0total人数
m4 = train_uafeature[train_uafeature.consumptionAbility==0][['aid']]
m4['consump_0_total']=1
m4 = m4.groupby('aid').agg('sum').reset_index()
##消费1total人数
m5 = train_uafeature[train_uafeature.consumptionAbility==1][['aid']]
m5['consump_1_total']=1
m5 = m5.groupby('aid').agg('sum').reset_index()
##消费2total人数
m6 = train_uafeature[train_uafeature.consumptionAbility==2][['aid']]
m6['consump_2_total']=1
m6 = m6.groupby('aid').agg('sum').reset_index()

for i in [m1,m2,m3,m4,m5,m6]:
    consumption_feature = pd.merge(consumption_feature,i,on='aid',how='left')

total_seed_num = consumption_feature.consump_0_seed+consumption_feature.consump_1_seed+consumption_feature.consump_2_seed
#consumption_feature['consump_0_seed_rate'] = consumption_feature.consump_0_seed.astype('float')/total_seed_num
consumption_feature['consump_1_seed_rate'] = consumption_feature.consump_1_seed.astype('float')/total_seed_num
consumption_feature['consump_2_seed_rate'] = consumption_feature.consump_2_seed.astype('float')/total_seed_num
consumption_feature['consump_0_convert_rate'] = consumption_feature.consump_0_seed.astype('float')/consumption_feature.consump_0_total
consumption_feature['consump_1_convert_rate'] = consumption_feature.consump_1_seed.astype('float')/consumption_feature.consump_1_total
consumption_feature['consump_2_convert_rate'] = consumption_feature.consump_2_seed.astype('float')/consumption_feature.consump_2_total

consumption_feature=consumption_feature[['aid','consump_0_seed_rate','consump_1_seed_rate','consump_2_seed_rate','consump_0_convert_rate','consump_1_convert_rate','consump_2_convert_rate']]
consumption_feature.to_csv('consumption_feature.csv',index=None)

del m1,m2,m3,m4,m5,m6,total_seed_num
gc.collect()

###########婚姻状况###############
user_feature.LBS.value_counts()


################是否有房###############
#train_uafeature.house.value_counts()
train_uafeature.house.replace(np.nan,0,inplace=True)
house_feature = train_uafeature[['aid']]
#有房seed人数
m1 = train_uafeature[(train_uafeature.house==1)&(train_uafeature.label==1)][['aid']]
m1['house_1_seed']=1
m1 = m1.groupby('aid').agg('sum').reset_index()
#无房seed人数
m2 = train_uafeature[(train_uafeature.house==0)&(train_uafeature.label==1)][['aid']]
m2['house_0_seed']=1
m2 = m2.groupby('aid').agg('sum').reset_index()
#有房总人数
m3 = train_uafeature[train_uafeature.house==1][['aid']]
m3['house_1_total']=1
m3 = m3.groupby('aid').agg('sum').reset_index()
#无房总人数
m4 = train_uafeature[train_uafeature.house==0][['aid']]
m4['house_0_total']=1
m4 = m4.groupby('aid').agg('sum').reset_index()

house_feature =pd.merge(house_feature,m1,on=['aid'],how='left')
house_feature =pd.merge(house_feature,m2,on=['aid'],how='left')
house_feature =pd.merge(house_feature,m3,on=['aid'],how='left')
house_feature =pd.merge(house_feature,m4,on=['aid'],how='left')
house_feature['total_seed_num'] = house_feature.house_1_seed+house_feature.house_0_seed
house_feature['house_rate']=house_feature.house_1_seed.astype('float')/house_feature.total_seed_num
house_feature['havehouse_convert_rate']=house_feature.house_1_seed.astype('float')/house_feature.house_1_total
house_feature['nohouse_convert_rate']=house_feature.house_0_seed.astype('float')/house_feature.house_0_total
house_feature=house_feature[['aid','house_rate','havehouse_convert_rate','nohouse_convert_rate']]
house_feature.to_csv('house_feature.csv',index=None)

del m1,m2,m3,m4,
gc.collect()

############广告对不同地理位置的转化率##############
lbs = list(train_adfeature.LBS.unique())
ad = list(train_adfeature.aid.unique())
lbs_feature = train_adfeature[['aid','LBS']]
for a in ad:
    m1 = train_adfeature[(train_adfeature.aid==a)&(train_adfeature.label==1)][['LBS']]
    m1['lbs_seed_total'+str(a)] = 1
    m1 = m1.groupby('LBS').agg('sum').reset_index()
    m1['aid'] = a
    m2 = train_adfeature[train_adfeature.aid==a][['LBS']]
    m2['lbs_total'+str(a)] = 1
    m2 = m2.groupby('LBS').agg('sum').reset_index()
    m2['aid'] = a
    lbs_feature = pd.merge(lbs_feature,m1,on=['aid','LBS'],how='left')
    lbs_feature = pd.merge(lbs_feature,m2,on=['aid','LBS'],how='left')
    lbs_feature['lbs_covert_rate'+str(a)] = lbs_feature['lbs_seed_total'+str(a)].astype('float')/lbs_feature['lbs_total'+str(a)]
    lbs_feature.drop(['lbs_total'+str(a),'lbs_seed_total'+str(a)],inplace=True,axis=1)
    


predict['label']=-1
data = pd.concat([train,predict])
data = pd.merge(data,ad_feature,on='aid',how='left')
data = pd.merge(data,user_feature,on='uid',how='left')

data.to_csv('data.csv',index=False)
##

data.house.replace(np.nan,0,inplace=True)
data=data.fillna('-1')

ad_feature.aid.value_counts()
ad_feature.advertiserId.value_counts()
ad_feature.campaignId.value_counts()
ad_feature.creativeId.value_counts()
ad_feature.productId.value_counts()



one_hot_feature=['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','marriageStatus','advertiserId','campaignId', 'creativeId',
       'adCategoryId', 'productId', 'productType']
vector_feature=['appIdAction','appIdInstall','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3']
for feature in one_hot_feature:
    try:
        data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
    except:
        data[feature] = LabelEncoder().fit_transform(data[feature])


        
