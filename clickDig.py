# -*- coding: utf-8 -*-

#提取和点击有关的特征

import numpy as np
import pandas as pd 

#V1-V2特征
#提取了用户点击总数，用户clickktime前的点击总数——n_count，n_all_count
#用户点击app总数，用户clickktime前的点击当前APP的总数-n_cl_app，n_all_cl_app
#用户是否转化过当前APP，is_convered

#统计appDetail（设了时间窗口，反而更差）
def clickDetail(x):
    
    n_count=0
    n_cl_app=0
    
    n_all_count=0
    n_all_cl_app=0
    
    is_convered=0
    
    c_time=int(str(x.clickTime))
    
    if not isinstance(x.cl_time,float):
        time_list=[int(i) for i in x.cl_time.strip().split(' ')]
        app_list=[int(i) for i in x.cl_app.strip().split(' ')]
        label_list=[int(i) for i in x.cl_label.strip().split(' ')]
        
        for i in range(len(time_list)):
            n_all_count+=1
            if x.appID==app_list[i]:
                n_all_cl_app+=1           
                
            if time_list[i]<c_time:
                n_count+=1
                if x.appID==app_list[i]:
                    n_cl_app+=1
                    
                if is_convered==0:                  
                    if app_list[i]==x.appID and label_list[i]==1:
                        is_convered=1                
    
    re=str(n_cl_app)+','+str(n_count)+','+str(n_all_count)+','+str(n_all_cl_app)
    
    re+=','+str(is_convered)
    
    print(re)
    
    return re
    
#挖掘统计信息
def getn_cl_app(x):
    return x.split(',')[0]

def getn_n_count(x):
    return x.split(',')[1]

def getn_all_count(x):
    return x.split(',')[2]

def getn_all_cl_app(x):
    return x.split(',')[3]

def get_is_convered(x):
    return x.split(',')[4]

def digClickDetail(data_type='train'):
    df_output=''
    save_path=''
    if data_type=='train':
        df_output=pd.read_csv('data/cutData/train_time_v1.csv')
        save_path='data/cutData/train_time_v2.csv'
    elif data_type=='test':
        df_output=pd.read_csv('data/first_merge/test_join_v1.csv')
        save_path='data/first_merge/test_join_v2.csv'
    else:
        print('data_type出错！')
        return
        
    df_detail=pd.read_csv('data/feature/click_detail_v1.csv')
    
    df_output=pd.merge(df_output,df_detail,how='left',on='userID')
    
    print('拼接,开始统计')
    
    del df_detail
        
    df_output['click_Detail']=df_output.apply(lambda x:clickDetail(x),axis=1)
    
    print('点击细节挖掘完成')  
    
    df_output['n_cl_app']=df_output['click_Detail'].apply(lambda x:getn_cl_app(x))
    print('点击同app次数统计完成')
    df_output['n_count']=df_output['click_Detail'].apply(lambda x:getn_n_count(x))
    print('点击同n_count次数统计完成')
    df_output['n_all_count']=df_output['click_Detail'].apply(lambda x:getn_all_count(x))
    print('点击同all_count次数统计完成')
    df_output['n_all_cl_app']=df_output['click_Detail'].apply(lambda x:getn_all_cl_app(x))
    print('n_all_cl_app统计完成')
    df_output['is_convered']=df_output['click_Detail'].apply(lambda x:get_is_convered(x))
    print('is_convered统计完成')
    
    df_output=df_output.drop(['click_Detail','cl_time','cl_app'],axis=1)
    
    df_output=df_output.drop(['cl_label'],axis=1)
    
    print('切割完成')
    
    df_output.to_csv(save_path,index=False) 
    
if __name__=='__main__':

    '''digClickDetail(data_type='train')
    digClickDetail(data_type='test')'''