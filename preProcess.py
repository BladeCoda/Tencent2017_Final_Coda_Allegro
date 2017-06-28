# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
  
#创建预处理文件的代码，提取特征之前需运行这段
  
#创建用户点击与安装记录
    
def create_AppAction():
    df_action=pd.read_csv('data/origin/user_app_actions.csv')       
    df_app_cat=pd.read_csv('data/origin/app_categories.csv')
    
    #拼接信息
    df_join=pd.merge(df_action,df_app_cat,how='left',on='appID') #拼接用户信息
    print('拼接完成')
    
    #记录总共的记录条数
    total_num=len(df_action)
    #获取所有的userID
    userID_list=df_action['userID'].values
    userID_list=list(set(userID_list))
    dict_time={}
    dict_app={}
    dict_cat={}
    #初始化字典
    for user in userID_list:
        dict_time[user]=''
        dict_app[user]=''
        dict_cat[user]=''
    print('已统计每个用户的app,开始统计各个用户最近安装的app')
    count=0
    for i,j,k,z in df_join.values:
        dict_time[i]+=(str(j)+' ')
        dict_app[i]+=(str(k)+' ')
        #统计type
        ca=str(z)
        if ca=='':
            ca='0'
        dict_cat[i]+=(ca+' ')
        count+=1
        if count%50000==0:
            print('字典已加载: %.2f %%'%(count/total_num*100))
    print('统计完成，开始保存')
    
    #下面的代码待定,用于保存。
    userID_list=list(dict_time.keys())
    app_time=list(dict_time.values())
    app_action=list(dict_app.values())
    app_type=list(dict_cat.values())
    
    df_list=pd.DataFrame({'userID':userID_list,
                          'timeList':app_time,
                          'actionList':app_action,
                          'action_type':app_type}) 
    df_list.to_csv('data/feature/action_list_v1.csv',index=False)
    
def create_CountDig():
    df_click=pd.read_csv('data/origin/train.csv')     
    df_click_t=pd.read_csv('data/origin/test.csv')
    df_user=pd.read_csv('data/origin/user.csv')
    
    df_ad=pd.read_csv('data/origin/ad.csv')
    df_click=pd.merge(df_click,df_ad,how='left',on='creativeID')
    df_click_t=pd.merge(df_click_t,df_ad,how='left',on='creativeID')
    
    del df_ad
    
    #获取所有的userID
    userID_list=df_user['userID'].values
    userID_list=list(set(userID_list))
    
    dic_app={}
    dic_time={}

    dic_label={}

    #初始化字典
    for user in userID_list:
        dic_app[user]=''
        dic_time[user]=''
        
        dic_label[user]=''

    print('已统计每个用户的app,开始统计各种安装的app列表')
    
    list_user=np.concatenate((df_click['userID'].values,df_click_t['userID'].values),axis=0)
    list_time=np.concatenate((df_click['clickTime'].values,df_click_t['clickTime'].values),axis=0)
    list_app=np.concatenate((df_click['appID'].values,df_click_t['appID'].values),axis=0)

    list_label=np.concatenate((df_click['label'].values,df_click_t['label'].values),axis=0)
    
    total=len(list_time)
    for i in range(total):
        dic_app[list_user[i]]+=(str(list_app[i])+' ')
        dic_time[list_user[i]]+=(str(list_time[i])+' ')
        
        dic_label[list_user[i]]+=(str(list_label[i])+' ')
    
        if i%50000==0:
            print('字典已加载: %.2f %%'%(i/total*100))

    
    #下面的代码待定,用于保存。creative
    userID_list=list(dic_app.keys())
    
    cl_app=list(dic_app.values())
    cl_time=list(dic_time.values())
    
    cl_label=list(dic_label.values())
    
    
    df_list=pd.DataFrame({'userID':userID_list,
                          'cl_time':cl_time,
                          'cl_app':cl_app,
                          'cl_label':cl_label}) 
    
    df_list.to_csv('data/feature/click_detail_v1.csv',index=False)  
          
def create_CountDigV2():
    df_click=pd.read_csv('data/origin/train.csv')     
    df_click_t=pd.read_csv('data/origin/test.csv')
    
    df_ad=pd.read_csv('data/origin/ad.csv')
    df_click=pd.merge(df_click,df_ad,how='left',on='creativeID')
    df_click_t=pd.merge(df_click_t,df_ad,how='left',on='creativeID')
    del df_ad 
    
    df_user=pd.read_csv('data/origin/user.csv')
    
    #获取所有的userID
    userID_list=df_user['userID'].values
    userID_list=list(set(userID_list))
    
    dic_cre={}
    dic_pos={}
    dic_time={}

    #初始化字典
    for user in userID_list:
        dic_cre[user]=''
        dic_pos[user]=''
        dic_time[user]=''

    print('已统计每个用户的app,开始统计各种安装的app列表')
    
    list_user=np.concatenate((df_click['userID'].values,df_click_t['userID'].values),axis=0)
    list_time=np.concatenate((df_click['clickTime'].values,df_click_t['clickTime'].values),axis=0)
    list_pos=np.concatenate((df_click['positionID'].values,df_click_t['positionID'].values),axis=0)
    list_cre=np.concatenate((df_click['creativeID'].values,df_click_t['creativeID'].values),axis=0)
    
    del df_click
    del df_click_t
    
    total=len(list_time)
    for i in range(total):
        dic_pos[list_user[i]]+=(str(list_pos[i])+' ')
        dic_time[list_user[i]]+=(str(list_time[i])+' ')
        dic_cre[list_user[i]]+=(str(list_cre[i])+' ')
    
        if i%50000==0:
            print('字典已加载: %.2f %%'%(i/total*100))

    
    #下面的代码待定,用于保存。
    userID_list=list(dic_pos.keys())
    
    cl_pos=list(dic_pos.values())
    cl_time=list(dic_time.values())
    cl_cre=list(dic_cre.values())
    
    df_list=pd.DataFrame({'userID':userID_list,
                          'cl_time':cl_time,
                          'cl_pos':cl_pos,
                          'cl_cre':cl_cre}) 
    
    df_list.to_csv('data/feature/click_detail_v2.csv',index=False) 
        
if __name__=='__main__':
    create_AppAction()
    create_CountDig()