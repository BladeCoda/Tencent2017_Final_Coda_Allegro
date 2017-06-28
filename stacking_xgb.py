# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.utils import shuffle
  
#训练分类器XGB
def trainClassifierXGB(x_train,y_train):
    #用30W条数据测试下（只要是调参）
    
    print('使用XGBOOST进行训练')
    dtrain=xgb.DMatrix(x_train,label=y_train)
    
    param = {'max_depth':7,'eta':0.25,'min_child_weight':1, 
             'silent':1, 'subsample':1,'colsample_bytree':1,
             'gamma':0,'scale_pos_weight':1,'lambda':50,
             'objective':'binary:logistic'}
             
    #不用设置线程，XGboost会自行设置所有的
    #param['nthread'] = 8

    plst = list(param.items())
    plst += [('eval_metric', 'auc')] # Multiple evals can be handled in this way 
    plst += [('eval_metric', 'ams@0')]
    
    num_round = 200
    bst = xgb.train(plst,dtrain,num_boost_round=num_round)
    
    print(param)
    print(num_round)
    
    return bst

def cutData():
    df_all=pd.read_csv('data/cutData/train_time_v9.csv')
    
    df_2=pd.read_csv('day30/data/train30_v9.csv') 
    print('开始拼接')
    df_all=df_all.append(df_2)
    del df_2
    
    df_all = shuffle(df_all,random_state=42)  
    
    step=len(df_all)//5
    
    train1=df_all[0:step]
    train2=df_all[step:2*step]
    train3=df_all[2*step:3*step]
    train4=df_all[3*step:4*step]
    train5=df_all[4*step:]

    del df_all
    return train1,train2,train3,train4,train5
    
def stacking_test():
    df_all=pd.read_csv('data/cutData/train_time_v9.csv')
    
    df_2=pd.read_csv('day30/data/train30_v9.csv') 
    print('开始拼接')
    df_all=df_all.append(df_2)
    del df_2
    
    feature_all=df_all.drop(['sort','label','clickTime','conversionTime',
                             'residence','appCategory'],axis=1).values
                             
    label_all=df_all['label'].values

    del df_all
    
    lgbm=trainClassifierXGB(feature_all,label_all)
    
    df_all_test=pd.read_csv('data/first_merge/test_join_v9.csv')
    feature_all=df_all_test.drop(['label','clickTime','instanceID',
                                  'residence','appCategory'],axis=1).values
    instanceID=df_all_test.instanceID.values
    dtest=xgb.DMatrix(feature_all)
                                  
    del df_all_test
                             
    prob = lgbm.predict(dtest)  
    
    df_prob=pd.DataFrame({'instanceID':instanceID,'xgb_prob':prob})  
    df_prob.to_csv('data/stacking/prob_xgb_test.csv',index=False) 
    
    
def stacking_train():
    
    train1,train2,train3,train4,train5=cutData()
    
    print('分割完成')
    
    list_sort=np.concatenate((train1['sort'].values,train2['sort'].values,
                              train3['sort'].values,train4['sort'].values,
                              train5['sort'].values))
    list_prob=[]
    #训练
    print('开始stacking1')
    strain=train2.append([train3,train4,train5])
    stest=train1 
    label_all=strain['label'].values
    feature_all=strain.drop(['sort','label','clickTime','conversionTime',
                             'residence','appCategory'],axis=1).values  
    del strain
    lgbm=trainClassifierXGB(feature_all,label_all)
    del feature_all
    del label_all
    test_all=stest.drop(['sort','label','clickTime','conversionTime',
                         'residence','appCategory'],axis=1).values 
    del stest
    dtest=xgb.DMatrix(test_all)
    del test_all
    prob = lgbm.predict(dtest)
    list_prob+=prob.tolist()
    
    print(len(list_prob))
    del dtest
    del prob
    del lgbm
    
    print('开始stacking2')
    strain=train1.append([train3,train4,train5])
    stest=train2 
    label_all=strain['label'].values
    feature_all=strain.drop(['sort','label','clickTime','conversionTime',
                             'residence','appCategory'],axis=1).values  
    del strain
    lgbm=trainClassifierXGB(feature_all,label_all)
    del feature_all
    del label_all
    test_all=stest.drop(['sort','label','clickTime','conversionTime',
                         'residence','appCategory'],axis=1).values 
    del stest
    dtest=xgb.DMatrix(test_all)
    del test_all
    prob = lgbm.predict(dtest)
    list_prob+=prob.tolist()
    
    print(len(list_prob))
    del dtest
    del prob
    del lgbm
    
    print('开始stacking3')
    strain=train1.append([train2,train4,train5])
    stest=train3 
    label_all=strain['label'].values
    feature_all=strain.drop(['sort','label','clickTime','conversionTime',
                             'residence','appCategory'],axis=1).values  
    del strain
    lgbm=trainClassifierXGB(feature_all,label_all)
    del feature_all
    del label_all
    test_all=stest.drop(['sort','label','clickTime','conversionTime',
                         'residence','appCategory'],axis=1).values 
    del stest
    dtest=xgb.DMatrix(test_all)
    del test_all
    prob = lgbm.predict(dtest)
    list_prob+=prob.tolist()
    
    print(len(list_prob))
    del dtest
    del prob
    del lgbm
    
    print('开始stacking4')
    strain=train1.append([train2,train3,train5])
    stest=train4
    label_all=strain['label'].values
    feature_all=strain.drop(['sort','label','clickTime','conversionTime',
                             'residence','appCategory'],axis=1).values  
    del strain
    lgbm=trainClassifierXGB(feature_all,label_all)
    del feature_all
    del label_all
    test_all=stest.drop(['sort','label','clickTime','conversionTime',
                         'residence','appCategory'],axis=1).values 
    del stest
    dtest=xgb.DMatrix(test_all)
    del test_all
    prob = lgbm.predict(dtest)
    list_prob+=prob.tolist()
    
    print(len(list_prob))
    del dtest
    del prob
    del lgbm
    
    print('开始stacking5')
    strain=train1.append([train2,train3,train4])
    stest=train5 
    label_all=strain['label'].values
    feature_all=strain.drop(['sort','label','clickTime','conversionTime',
                             'residence','appCategory'],axis=1).values  
    del strain
    lgbm=trainClassifierXGB(feature_all,label_all)
    del feature_all
    del label_all
    test_all=stest.drop(['sort','label','clickTime','conversionTime',
                         'residence','appCategory'],axis=1).values 
    del stest
    dtest=xgb.DMatrix(test_all)
    del test_all
    prob = lgbm.predict(dtest)
    list_prob+=prob.tolist()
    
    print(len(list_prob))
    del dtest
    del prob
    del lgbm
    
    df_prob=pd.DataFrame({'sort':list_sort,'xgb_prob':list_prob})
    
    df_prob=df_prob.sort(columns='sort')
    
    del train1
    del train2
    del train3
    del train4
    del train5
    
    df_prob.to_csv('data/stacking/prob_xgb_train.csv',index=False) 

if __name__=='__main__':
    #stacking_train()
    stacking_test()