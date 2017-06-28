# -*- coding: utf-8 -*-

#挖掘转化率平滑的代码

import scipy.special as special
from collections import Counter
import pandas as pd 
class BayesianSmoothing(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
    def update(self, imps, clks, iter_num, epsilon):
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(imps, clks, self.alpha, self.beta)
            if abs(new_alpha-self.alpha)<epsilon and abs(new_beta-self.beta)<epsilon:
                break
            print(new_alpha,new_beta,i)
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, imps, clks, alpha, beta):
        numerator_alpha = 0.0
        numerator_beta = 0.0
        denominator = 0.0
        for i in range(len(imps)):
            numerator_alpha += (special.digamma(clks[i]+alpha) - special.digamma(alpha))
            numerator_beta += (special.digamma(imps[i]-clks[i]+beta) - special.digamma(beta))
            denominator += (special.digamma(imps[i]+alpha+beta) - special.digamma(alpha+beta))

        return alpha*(numerator_alpha/denominator), beta*(numerator_beta/denominator)
    
def PHpos():    
    df_pos=pd.read_csv('data/origin/position.csv')    
    pos_all_list=list(set(df_pos.positionID.values)) 
    del df_pos    
    print('载入完成，开始拼接')    
    df_train=pd.read_csv('data/origin/train.csv')
    
    #-------------------------------------
    print('开始统计pos平滑')
    bs = BayesianSmoothing(1, 1)    
    dic_i=dict(Counter(df_train.positionID.values))
    dic_cov=dict(Counter(df_train[df_train['label']==1].positionID.values))  
    l=list(set(df_train.positionID.values))     
    I=[]
    C=[]
    for posID in l:
        I.append(dic_i[posID])
    for posID in l:
        if posID not in dic_cov:
            C.append(0)
        else:
            C.append(dic_cov[posID])        
    print('开始平滑操作')           
    bs.update(I, C, 100000, 0.0000000001)
    print(bs.alpha, bs.beta)  
    print('构建平滑转化率')
    dic_PH={}
    for pos in pos_all_list:
        if pos not in dic_i:
            dic_PH[pos]=(bs.alpha)/(bs.alpha+bs.beta)
        elif pos not in dic_cov:
            dic_PH[pos]=(bs.alpha)/(dic_i[pos]+bs.alpha+bs.beta)
        else:
            dic_PH[pos]=(dic_cov[pos]+bs.alpha)/(dic_i[pos]+bs.alpha+bs.beta)   
    df_out=pd.DataFrame({'positionID':list(dic_PH.keys()),
                         'PH_pos':list(dic_PH.values())})
    
    df_out.to_csv('data/feature/PL_pos.csv',index=False)

    print('开始复制保存')
    
def PHapp():    
    df_ad=pd.read_csv('data/origin/ad.csv')
    app_all_list=list(set(df_ad.appID.values))   
  
    print('载入完成，开始拼接')    
    df_train=pd.read_csv('data/origin/train.csv')
    df_train=pd.merge(df_train,df_ad,how='left',on='creativeID')    
    del df_ad
    
    #-------------------------------------
    print('开始统计app平滑')
    bs = BayesianSmoothing(1, 1)    
    dic_i=dict(Counter(df_train.appID.values))
    dic_cov=dict(Counter(df_train[df_train['label']==1].appID.values))  
    l=list(set(df_train.appID.values))     
    I=[]
    C=[]
    for appID in l:
        I.append(dic_i[appID])
    for appID in l:
        if appID not in dic_cov:
            C.append(0)
        else:
            C.append(dic_cov[appID])        
    print('开始平滑操作')           
    bs.update(I, C, 100000, 0.0000000001)
    print(bs.alpha, bs.beta)  
    print('构建平滑转化率')
    dic_PH={}
    for app in app_all_list:
        if app not in dic_i:
            dic_PH[app]=(bs.alpha)/(bs.alpha+bs.beta)
        elif app not in dic_cov:
            dic_PH[app]=(bs.alpha)/(dic_i[app]+bs.alpha+bs.beta)
        else:
            dic_PH[app]=(dic_cov[app]+bs.alpha)/(dic_i[app]+bs.alpha+bs.beta)   
    df_out=pd.DataFrame({'appID':list(dic_PH.keys()),
                         'PH_app':list(dic_PH.values())})
    
    df_out.to_csv('data/feature/PL_app.csv',index=False)

    print('开始复制保存')
    
def PHcre():    
    df_ad=pd.read_csv('data/origin/ad.csv')
    cre_all_list=list(set(df_ad.creativeID.values))   
  
    print('载入完成，开始拼接')    
    df_train=pd.read_csv('data/origin/train.csv')
    df_train=pd.merge(df_train,df_ad,how='left',on='creativeID')    
    del df_ad
    
    #-------------------------------------
    print('开始统计app平滑')
    bs = BayesianSmoothing(1, 1)    
    dic_i=dict(Counter(df_train.creativeID.values))
    dic_cov=dict(Counter(df_train[df_train['label']==1].creativeID.values))  
    l=list(set(df_train.creativeID.values))     
    I=[]
    C=[]
    for creID in l:
        I.append(dic_i[creID])
    for creID in l:
        if creID not in dic_cov:
            C.append(0)
        else:
            C.append(dic_cov[creID])        
    print('开始平滑操作')           
    bs.update(I, C, 100000, 0.0000000001)
    print(bs.alpha, bs.beta)  
    print('构建平滑转化率')
    dic_PH={}
    for cre in cre_all_list:
        if cre not in dic_i:
            dic_PH[cre]=(bs.alpha)/(bs.alpha+bs.beta)
        elif cre not in dic_cov:
            dic_PH[cre]=(bs.alpha)/(dic_i[cre]+bs.alpha+bs.beta)
        else:
            dic_PH[cre]=(dic_cov[cre]+bs.alpha)/(dic_i[cre]+bs.alpha+bs.beta)   
    df_out=pd.DataFrame({'creativeID':list(dic_PH.keys()),
                         'PH_cre':list(dic_PH.values())})
    
    df_out.to_csv('data/feature/PL_cre.csv',index=False)

    print('开始复制保存')
    
def PHuser():    
    df_user=pd.read_csv('data/origin/user.csv')
    user_all_list=list(set(df_user.userID.values))   
    del df_user
    
    df_train=pd.read_csv('data/origin/train.csv')
    df_train=df_train[(df_train.clickTime//1000000<28)&(df_train.clickTime//1000000>=26)]
    
    #-------------------------------------
    print('开始统计user平滑')
    bs = BayesianSmoothing(1, 1)    
    dic_i=dict(Counter(df_train.userID.values))
    dic_cov=dict(Counter(df_train[df_train['label']==1].userID.values))  
    l=list(set(df_train.userID.values))     
    I=[]
    C=[]
    for userID in l:
        I.append(dic_i[userID])
    for userID in l:
        if userID not in dic_cov:
            C.append(0)
        else:
            C.append(dic_cov[userID])        
    print('开始平滑操作')           
    bs.update(I, C, 25, 0.0000000001)
    print(bs.alpha, bs.beta)  
    
    print('读取全部数据')
    
    df_train=pd.read_csv('data/origin/train.csv')
    df_train=df_train[(df_train.clickTime//1000000<28)]
    dic_i=dict(Counter(df_train.userID.values))
    dic_cov=dict(Counter(df_train[df_train['label']==1].userID.values)) 
    
    print('构建平滑转化率')
    
    dic_PH={}
    for user in user_all_list:
        if user not in dic_i:
            dic_PH[user]=(bs.alpha)/(bs.alpha+bs.beta)
        elif user not in dic_cov:
            dic_PH[user]=(bs.alpha)/(dic_i[user]+bs.alpha+bs.beta)
        else:
            dic_PH[user]=(dic_cov[user]+bs.alpha)/(dic_i[user]+bs.alpha+bs.beta)   
    df_out=pd.DataFrame({'userID':list(dic_PH.keys()),
                         'PH_user':list(dic_PH.values())})
    
    df_out.to_csv('data/feature/PL_user.csv',index=False)

    print('开始复制保存')
    
#V3-V4特征
#挖掘pos,cre,app的平滑转化率

def digPHconveseRate(data_type='train'):
    df_output=''
    save_path=''
    if data_type=='train':
        df_output=pd.read_csv('data/cutData/train_time_v3.csv')
        save_path='data/cutData/train_time_v4.csv'
    elif data_type=='test':
        df_output=pd.read_csv('data/first_merge/test_join_v3.csv')
        save_path='data/first_merge/test_join_v4.csv'
    else:
        print('data_type出错！')
        return
        
    df_appPH=pd.read_csv('data/feature/PL_app.csv')
    df_output=pd.merge(df_output,df_appPH,how='left',on='appID')
    del df_appPH
   
    df_posPH=pd.read_csv('data/feature/PL_pos.csv')
    df_output=pd.merge(df_output,df_posPH,how='left',on='positionID')
    del df_posPH
   
    df_crePH=pd.read_csv('data/feature/PL_cre.csv')
    df_output=pd.merge(df_output,df_crePH,how='left',on='creativeID')
    del df_crePH
    
    print('保存中......')
    
    df_output.to_csv(save_path,index=False) 
    
#V7-V8特征
#挖掘user的平滑转化率

def digPHconveseRateV2(data_type='train'):
    df_output=''
    save_path=''
    if data_type=='train':
        df_output=pd.read_csv('data/cutData/train_time_v7.csv')
        save_path='data/cutData/train_time_v8.csv'
    elif data_type=='test':
        df_output=pd.read_csv('data/first_merge/test_join_v7.csv')
        save_path='data/first_merge/test_join_v8.csv'
    else:
        print('data_type出错！')
        return
        
    df_userPH=pd.read_csv('data/feature/PL_user.csv')
    df_output=pd.merge(df_output,df_userPH,how='left',on='userID')
    del df_userPH
    
    print('保存中......')
    
    df_output.to_csv(save_path,index=False) 
    

if __name__ == '__main__':
    #PHpos()
    #PHapp()
    #PHcre()
    #PHuser()
    
    #digPHconveseRate(data_type='train')
    #digPHconveseRate(data_type='test')
    
    digPHconveseRateV2(data_type='train')
    digPHconveseRateV2(data_type='test')