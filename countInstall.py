# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

#创建install统计文件的代码，切割后统计

def cut_install():
    df_installed=pd.read_csv('data/origin/user_installedapps.csv')  
    df_app_cat=pd.read_csv('data/origin/app_categories.csv')
    print('加载完成')
    total=len(df_installed)
    p_len=total//10
    for i in range(9):
        print('开始切割第%d部分'%(i+1))
        df_part=df_installed[i*p_len:(i+1)*p_len]
        df_part=pd.merge(df_part,df_app_cat,how='left',on='appID')
        p_name='data/feature/install_cut/cut_p'+str(i+1)+'.csv'
        df_part.to_csv(p_name,index=False)
        del df_part
    print('开始切割最后部分')
    df_part=df_installed[9*p_len:]
    df_part=pd.merge(df_part,df_app_cat,how='left',on='appID')
    df_part.to_csv('data/feature/install_cut/cut_p10.csv',index=False)
    del df_part
        

def create_AppInstalled():
    
    df_user=pd.read_csv('data/origin/user.csv')
    user_list=df_user['userID'].values
    dict_list={}
    dict_cat={}
    #初始化字典
    for user in user_list:
        dict_list[user]=''
        dict_cat[user]=''
    print('初始化完成,开始统计各种安装的app列表')
    del df_user
    del user_list

    for i in range(10):
        print('开始统计第%d份数据'%(i+1))
        path='data/feature/install_cut/cut_p'+str(i+1)+'.csv'
        #记录总共的记录条数
        df_installed=pd.read_csv(path)
        total_num=len(df_installed)
        #获取所有的userID
        userID_list=df_installed['userID'].values
        userID_list=list(set(userID_list))
        feature_all=df_installed.values
        del df_installed
        
        count=0
        for i,j,k in feature_all:
            dict_list[i]+=(str(j)+' ')
            #统计type
            ca=str(k)
            if ca=='':
                ca='0'
            dict_cat[i]+=(ca+' ')
            count+=1
            if count%100000==0:
                print('字典已加载: %.2f %%'%(count/total_num*100))
        
        del feature_all
    
    #下面的代码待定,用于保存。
    userID_list=list(dict_list.keys())
    app_install=list(dict_list.values())
    app_type=list(dict_cat.values())
    df_list=pd.DataFrame({'userID':userID_list,
                          'appList':app_install,
                          'install_type':app_type}) 
    df_list.to_csv('data/feature/installed_list_v1.csv',index=False)
    
if __name__=='__main__':
    #cut_install()
    #create_AppInstalled()