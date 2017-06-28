# -*- coding: utf-8 -*-

import scipy.special as special
from collections import Counter
import pandas as pd 

    
#V3-V4特征
def digPHconveseRate():
    df_output=pd.read_csv('data/train30_v3.csv')
    save_path='data/train30_v4.csv'
        
    df_appPH=pd.read_csv('../data/feature/PL_app.csv')
    df_output=pd.merge(df_output,df_appPH,how='left',on='appID')
    del df_appPH
   
    df_posPH=pd.read_csv('../data/feature/PL_pos.csv')
    df_output=pd.merge(df_output,df_posPH,how='left',on='positionID')
    del df_posPH
   
    df_crePH=pd.read_csv('../data/feature/PL_cre.csv')
    df_output=pd.merge(df_output,df_crePH,how='left',on='creativeID')
    del df_crePH
    
    print('保存中......')
    
    df_output.to_csv(save_path,index=False) 
    
def digPHconveseRateV2():
    df_output=pd.read_csv('data/train30_v7.csv')
    save_path='data/train30_v8.csv'
        
    df_userPH=pd.read_csv('../data/feature/PL_user.csv')
    df_output=pd.merge(df_output,df_userPH,how='left',on='userID')
    del df_userPH
    
    print('保存中......')
    
    df_output.to_csv(save_path,index=False) 
    

if __name__ == '__main__':
    #digPHconveseRate()
    digPHconveseRateV2()