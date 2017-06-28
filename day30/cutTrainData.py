# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.utils import shuffle

def cutMainInfo(x):
    Main=0
    if x<10:
        Main=x
    else:
        Main=x//100
    return Main
    
def cutSubInfo(x):
    Sub=0
    if x<10:
        Sub=0
    else:
        Sub=x%100
    return Sub  
    
def cutTrainByTime():
    df_all=pd.read_csv('../data/origin/train.csv') 
    print('原始数据读取完成')
    
    df_all['sort']=[i for i in range(len(df_all))]
    
    df_out=df_all[(df_all.clickTime//1000000==30)]
        
    print('切割完成，开始排序')       
    df_out=df_out.sort(columns='clickTime')
    print('排序完成，开始拼接')
    
    df_ad=pd.read_csv('../data/origin/ad.csv')
    df_app_cat=pd.read_csv('../data/origin/app_categories.csv')
    df_pos=pd.read_csv('../data/origin/position.csv')
    df_user=pd.read_csv('../data/origin/user.csv')
    
    #拼接信息
    df_join=pd.merge(df_out,df_user,how='left',on='userID') #拼接用户信息
    del df_out
    del df_user
    print('user拼接完成')
    
    df_join=pd.merge(df_join,df_pos,how='left',on='positionID')#拼接position信息
    del df_pos
    print('pos拼接完成')

    df_join=pd.merge(df_join,df_ad,how='left',on='creativeID')#拼接广告素材信息
    del df_ad
    print('ad拼接完成')

    df_join=pd.merge(df_join,df_app_cat,how='left',on='appID')#拼接广告素材信息
    del df_app_cat
    print('app拼接完成')
    
    #将appTpye切割成两个维度
    df_join['appTypeMain']=df_join['appCategory'].apply(lambda x:cutMainInfo(x))
    df_join['appTypeSub']=df_join['appCategory'].apply(lambda x:cutSubInfo(x))
    
    df_join['home_m']=df_join['hometown'].apply(lambda x:cutMainInfo(x))
    df_join['home_s']=df_join['hometown'].apply(lambda x:cutSubInfo(x))
    
    df_join['hour']=df_join['clickTime'].apply(lambda x:(x%1000000)//10000)
    df_join['minute']=df_join['clickTime'].apply(lambda x:(x%10000)//100)
    
    #去掉一些没用的信息

    print('拼接完成')
    df_join.to_csv('data/train30_v1.csv',index=False)
    print('保存完成')
    
    del df_join  

if __name__=='__main__':
    cutTrainByTime()