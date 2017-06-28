# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.utils import shuffle

#切割训练数据的代码，比赛模型使用的是按照时间切割cutTrainByTime

def cutTrain():
    df_all=pd.read_csv('data/origin/train.csv') 
    print('原始数据读取完成')
    flag=0
    
    df_out=[]
    
    for i in range(17,31):   
        print('切割第',i,'天的数据')
        df_t_1=df_all[(df_all.clickTime//1000000==i)&(df_all.label==1)]
        df_t_0=df_all[(df_all.clickTime//1000000==i)&(df_all.label==0)]

        total1=len(df_t_1)//8
        total0=len(df_t_0)//8

        df_t_1 = shuffle(df_t_1,random_state=42) 
        df_t_0 = shuffle(df_t_0,random_state=42)  
        
        if flag==0:
            flag=1
            df_out=df_t_0[:total0]
            df_out=df_out.append(df_t_1[:total1]) 
        else:
            df_out=df_out.append(df_t_0[:total0])   
            df_out=df_out.append(df_t_1[:total1]) 
        del df_t_0
        del df_t_1
        
    print('切割完成，开始排序')       
    df_out=df_out.sort(columns='clickTime')
    print('排序完成，开始保存')
    df_out.to_csv('data/cutData/train_cut_v1.csv',index=False)
    
def cutTrainByTime():
    df_all=pd.read_csv('data/origin/train.csv') 
    print('原始数据读取完成')
    
    df_all['sort']=[i for i in range(len(df_all))]
    
    df_out=df_all[(df_all.clickTime//1000000==28)]
    df_out=df_out.append(df_all[(df_all.clickTime//1000000==29)])
    
    df_out2=df_all[(df_all.clickTime//1000000==30)]
        
    print('切割完成，开始排序')       
    df_out=df_out.sort(columns='clickTime')
    df_out2=df_out2.sort(columns='clickTime')
    print('排序完成，开始拼接')
    
    df_ad=pd.read_csv('data/origin/ad.csv')
    df_app_cat=pd.read_csv('data/origin/app_categories.csv')
    df_pos=pd.read_csv('data/origin/position.csv')
    df_user=pd.read_csv('data/origin/user.csv')
    
    #拼接信息
    df_join=pd.merge(df_out,df_user,how='left',on='userID') #拼接用户信息
    df_join2=pd.merge(df_out2,df_user,how='left',on='userID') #拼接用户信息
    del df_out
    del df_user
    del df_out2
    print('user拼接完成')
    
    df_join=pd.merge(df_join,df_pos,how='left',on='positionID')#拼接position信息
    df_join2=pd.merge(df_join2,df_pos,how='left',on='positionID')#拼接position信息
    del df_pos
    print('pos拼接完成')

    df_join=pd.merge(df_join,df_ad,how='left',on='creativeID')#拼接广告素材信息
    df_join2=pd.merge(df_join2,df_ad,how='left',on='creativeID')#拼接广告素材信息
    del df_ad
    print('ad拼接完成')

    df_join=pd.merge(df_join,df_app_cat,how='left',on='appID')#拼接广告素材信息
    df_join2=pd.merge(df_join2,df_app_cat,how='left',on='appID')#拼接广告素材信息
    del df_app_cat
    print('app拼接完成')
    
    #将appTpye切割成两个维度
    df_join['appTypeMain']=df_join['appCategory'].apply(lambda x:cutMainInfo(x))
    df_join['appTypeSub']=df_join['appCategory'].apply(lambda x:cutSubInfo(x))
    
    df_join['home_m']=df_join['hometown'].apply(lambda x:cutMainInfo(x))
    df_join['home_s']=df_join['hometown'].apply(lambda x:cutSubInfo(x))
    
    df_join2['appTypeMain']=df_join2['appCategory'].apply(lambda x:cutMainInfo(x))
    df_join2['appTypeSub']=df_join2['appCategory'].apply(lambda x:cutSubInfo(x))
    
    df_join2['home_m']=df_join2['hometown'].apply(lambda x:cutMainInfo(x))
    df_join2['home_s']=df_join2['hometown'].apply(lambda x:cutSubInfo(x))
    
    df_join['hour']=df_join['clickTime'].apply(lambda x:(x%1000000)//10000)
    df_join['minute']=df_join['clickTime'].apply(lambda x:(x%10000)//100)
    
    df_join2['hour']=df_join2['clickTime'].apply(lambda x:(x%1000000)//10000)
    df_join2['minute']=df_join2['clickTime'].apply(lambda x:(x%10000)//100)
    
    #去掉一些没用的信息

    print('拼接完成')
    df_join.to_csv('data/cutData/train_time_v1.csv',index=False)
    print('保存完成')
    df_join2.to_csv('data/cutData/train_time_validate.csv',index=False)
    print('保存完成')
    
    del df_join  
    
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

def join_cut():
    df_ad=pd.read_csv('data/origin/ad.csv')
    df_app_cat=pd.read_csv('data/origin/app_categories.csv')
    df_pos=pd.read_csv('data/origin/position.csv')
    df_train=pd.read_csv('data/cutData/train_cut_v1.csv')
    df_user=pd.read_csv('data/origin/user.csv')
    
    print('读取完成开始拼接')
    
    #拼接信息
    df_join=pd.merge(df_train,df_user,how='left',on='userID') #拼接用户信息
    del df_train
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
    df_join.to_csv('data/cutData/train_join_v1.csv',index=False)
    print('保存完成')
    
    del df_join    

if __name__=='__main__':
    cutTrainByTime()