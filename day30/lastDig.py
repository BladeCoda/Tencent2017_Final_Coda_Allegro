# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def lastDetail(x):
    
    n_cl_cre=0
    n_all_cl_cre=0
    n_cl_pos=0
    n_all_cl_pos=0
    
    c_time=int(str(x.clickTime))
    
    if not isinstance(x.cl_time,float):
        time_list=[int(i) for i in x.cl_time.strip().split(' ')]
        cre_list=[int(i) for i in x.cl_cre.strip().split(' ')]
        pos_list=[int(i) for i in x.cl_pos.strip().split(' ')]
        
        for i in range(len(time_list)):
            if x.creativeID==cre_list[i]:
                n_all_cl_cre+=1      
            if x.positionID==pos_list[i]:
                n_all_cl_pos+=1  
                
            if time_list[i]<c_time:
                if x.creativeID==cre_list[i]:
                    n_cl_cre+=1 
                if x.positionID==pos_list[i]:
                    n_cl_pos+=1
                       
    re=str(n_cl_cre)+','+str(n_all_cl_cre)+','+str(n_cl_pos)+','+str(n_all_cl_pos)
    
    print(re)
    
    return re
    
def getn_cl_cre(x):
    return x.split(',')[0]

def getn_all_cl_cre(x):
    return x.split(',')[1]

def getn_cl_pos(x):
    return x.split(',')[2]

def getn_all_cl_pos(x):
    return x.split(',')[3]

    
def digLastDetail():
  
    in_path='data/train30_v8.csv'
    save_path='data/train30_v9.csv'
    df_all=pd.read_csv('data/train30_v1.csv')
    
    df_detail=pd.read_csv('../data/feature/click_detail_v2.csv')
    df_all=pd.merge(df_all,df_detail,how='left',on='userID')
    
    print('拼接,开始统计')
    
    del df_detail
        
    df_all['last_Detail']=df_all.apply(lambda x:lastDetail(x),axis=1)
    
    print('APP细节挖掘完成')
    
    df_all['n_cl_cre']=df_all.last_Detail.apply(lambda x:getn_cl_cre(x))
    print('n_cl_cre统计完成')
    df_all['n_all_cl_cre']=df_all.last_Detail.apply(lambda x:getn_all_cl_cre(x))
    print('n_all_cl_cre统计完成')
    df_all['n_cl_pos']=df_all.last_Detail.apply(lambda x:getn_cl_pos(x))
    print('n_cl_pos统计完成')
    df_all['n_all_cl_pos']=df_all.last_Detail.apply(lambda x:getn_all_cl_pos(x))
    print('n_all_cl_pos统计完成')
    
    n_cl_cre=df_all['n_cl_cre'].values
    n_all_cl_cre=df_all['n_all_cl_cre'].values
    n_cl_pos=df_all['n_cl_pos'].values
    n_all_cl_pos=df_all['n_all_cl_pos'].values

    del df_all
    
    print('开始合并') 
    df_output=pd.read_csv(in_path)
    
    df_output['n_cl_cre']=n_cl_cre
    df_output['n_all_cl_cre']=n_all_cl_cre
    df_output['n_cl_pos']=n_cl_pos
    df_output['n_all_cl_pos']=n_all_cl_pos

    print('开始保存')
    
    df_output.to_csv(save_path,index=False)
    
if __name__=='__main__':
    digLastDetail()