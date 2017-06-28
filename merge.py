# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#用于拼接基础信息的代码

import numpy as np
import pandas as pd
    
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

def join_train():
    df_ad=pd.read_csv('data/origin/ad.csv')
    df_app_cat=pd.read_csv('data/origin/app_categories.csv')
    df_pos=pd.read_csv('data/origin/position.csv')
    df_train=pd.read_csv('data/origin/train.csv')
    df_user=pd.read_csv('data/origin/user.csv')
    
    #拼接信息
    df_join=pd.merge(df_train,df_user,how='left',on='userID') #拼接用户信息
    del df_train
    del df_user
    
    df_join=pd.merge(df_join,df_pos,how='left',on='positionID')#拼接position信息
    del df_pos

    df_join=pd.merge(df_join,df_ad,how='left',on='creativeID')#拼接广告素材信息
    del df_ad

    df_join=pd.merge(df_join,df_app_cat,how='left',on='appID')#拼接广告素材信息
    del df_app_cat

    
    #将appTpye切割成两个维度
    df_join['appTypeMain']=df_join['appCategory'].apply(lambda x:cutMainInfo(x))
    df_join['appTypeSub']=df_join['appCategory'].apply(lambda x:cutSubInfo(x))
    
    df_join['home_m']=df_join['hometown'].apply(lambda x:cutMainInfo(x))
    df_join['home_s']=df_join['hometown'].apply(lambda x:cutSubInfo(x))
    
    df_join['hour']=df_join['clickTime'].apply(lambda x:(x%1000000)//10000)
    df_join['minute']=df_join['clickTime'].apply(lambda x:(x%10000)//100)
    
    #去掉一些没用的信息

    print('拼接完成')
    df_join.to_csv('data/first_merge/train_join_v1.csv',index=False)
    print('保存完成')
    
    del df_join    
    
def join_test():
    df_ad=pd.read_csv('data/origin/ad.csv')
    df_app_cat=pd.read_csv('data/origin/app_categories.csv')
    df_pos=pd.read_csv('data/origin/position.csv')
    df_test=pd.read_csv('data/origin/test.csv')
    df_user=pd.read_csv('data/origin/user.csv')
    
    #拼接信息
    df_join=pd.merge(df_test,df_user,how='left',on='userID') #拼接用户信息
    del df_user
    del df_test

    df_join=pd.merge(df_join,df_pos,how='left',on='positionID')#拼接position信息
    del df_pos

    df_join=pd.merge(df_join,df_ad,how='left',on='creativeID')#拼接广告素材信息
    del df_ad

    df_join=pd.merge(df_join,df_app_cat,how='left',on='appID')#拼接广告素材信息
    del df_app_cat

    
    #将appTpye切割成两个维度
    df_join['appTypeMain']=df_join['appCategory'].apply(lambda x:cutMainInfo(x))
    df_join['appTypeSub']=df_join['appCategory'].apply(lambda x:cutSubInfo(x))
    
    df_join['home_m']=df_join['hometown'].apply(lambda x:cutMainInfo(x))
    df_join['home_s']=df_join['hometown'].apply(lambda x:cutSubInfo(x))
    
    df_join['hour']=df_join['clickTime'].apply(lambda x:(x%1000000)//10000)
    df_join['minute']=df_join['clickTime'].apply(lambda x:(x%10000)//100)
    
    #---去掉一些没用的信息----

    print('拼接完成')
    df_join.to_csv('data/first_merge/test_join_v1.csv',index=False)
    print('保存完成')
    
    del df_join    
    
#主函数入口
if __name__=='__main__':
    
    '''
    #下面是一个简要的操作(备忘)
    df_ad_data=df_ad.get_values()#直接把csv转化成一个矩阵
    app_id=df_ad['appID']#直接获取appID,仍然是一个Series
    small=df_ad[0:3]#获取0,1,2三条记录
    df_sort=df_ad.sort_values(by='creativeID',ascending=True)#按照creativeID来排序creativeID
    #一个查找的例子
    #选择appPlatform=1,adID=1593的记录，并只获取'creativeID','adID'
    df_sel=df_ad[['creativeID','adID']][(df_ad.appPlatform==1)&(df_ad.adID==1593)]
    '''
    
    #join_train()
    join_test()
