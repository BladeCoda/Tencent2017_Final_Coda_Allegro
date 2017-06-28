# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.cross_validation import train_test_split

import scipy as sp

def logloss(act, pred):
  epsilon = 1e-15
  pred = sp.maximum(epsilon, pred)
  pred = sp.minimum(1-epsilon, pred)
  ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
  ll = ll * -1.0/len(act)
  return ll

def loadCSV(path=''):
    return pd.read_csv(path)
   
    
#训练分类器XGB
def trainClassifierLGBM(x_train,y_train):
    
    print('使用LIGHTBGM进行训练')
    
    lgb_train = lgb.Dataset(x_train, y_train)

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        #'max_depth':15,
        'num_leaves': 355,
        #'max_bin':150,  
        'learning_rate': 0.02,
        'feature_fraction': 0.5,
        #'bagging_fraction': 0.85,
        #'bagging_freq': 5,
        'verbose': 0,
    }
    
    #origin:learning_rate': 0.02，num_boost_round=700,'num_leaves': 355
    
    lgbm = lgb.train(params,
                lgb_train,
                num_boost_round=700)
    
    print(params)
    
    return lgbm
    
#encoding代表是不是有独热编码信息（需要分解）
def predict_test_prob(lgbm):
    df_all=loadCSV('data/first_merge/test_join_v9.csv') 
    
    df_sta_xgb=loadCSV('data/stacking/prob_xgb_test.csv') 
    print('开始拼接')
    df_all=pd.merge(df_all,df_sta_xgb,how='left',on='instanceID')
    del df_sta_xgb   
    instanceID=df_all.instanceID.values
    feature_all=df_all.drop(['label','clickTime','instanceID',
                             'residence','appCategory'],axis=1).values
                             
    prob = lgbm.predict(feature_all, num_iteration=lgbm.best_iteration)

    output=pd.DataFrame({'instanceID':instanceID,'prob':prob})
    
    output.to_csv('result/submission.csv',index=False) 

#交叉验证
def cross_validat(df_all,test_size=0.2):
    #0.097726
    #feature_all=df_all.drop(['userID','label','clickTime','conversionTime'],axis=1).values
    
    feature_all=df_all.drop(['sort','label','clickTime','conversionTime',
                             'residence','appCategory'],axis=1).values

    #测试V6各种特征
        
    label_all=df_all.label.values
    
    del df_all

    x_train,x_test,y_train,y_test=train_test_split(feature_all,label_all,test_size=test_size,random_state=42)
    print('数据集切割完成')
    
    del feature_all
    del label_all
    
    lgbm=trainClassifierLGBM(x_train,y_train)
    print('训练完成')
    
    prob = lgbm.predict(x_test, num_iteration=lgbm.best_iteration)
    
    loss=logloss(y_test,prob)
    print('交叉验证损失为:',loss)
    
    return prob
    
def pointReserve():
    df_point=loadCSV('result/submission.csv')
    df_point['prob']=df_point.prob.apply(lambda x:round(x,10))
    df_point.to_csv('result/submission2.csv',index=False) 
    
#主函数入口
if __name__=='__main__':
    
   df_all=loadCSV('data/cutData/train_time_v9.csv')
   
   df_2=loadCSV('day30/data/train30_v9.csv') 
   print('开始拼接')
   df_all=df_all.append(df_2)
   del df_2
   
   df_sta_xgb=loadCSV('data/stacking/prob_xgb_train.csv') 
   print('开始拼接')
   df_all=pd.merge(df_all,df_sta_xgb,how='left',on='sort')
   del df_sta_xgb
   
   print('开始训练')
   
   #pre=cross_validat(df_all,test_size=0.2)
   
   feature_all=df_all.drop(['sort','label','clickTime','conversionTime',
                             'residence','appCategory'],axis=1).values
   label_all=df_all.label.values
   
   del df_all
   
   bst=trainClassifierLGBM(feature_all,label_all)
   print('分类器训练完成')
   predict_test_prob(bst)
   print('结果预测完成')