# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.utils import shuffle

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
    
    lgbm=trainClassifierLGBM(feature_all,label_all)
    
    df_all_test=pd.read_csv('data/first_merge/test_join_v9.csv')
    feature_all=df_all_test.drop(['label','clickTime','instanceID',
                                  'residence','appCategory'],axis=1).values
    instanceID=df_all_test.instanceID.values
                                  
    del df_all_test
                             
    prob = lgbm.predict(feature_all, num_iteration=lgbm.best_iteration)
    
    df_prob=pd.DataFrame({'instanceID':instanceID,'lgbm_prob':prob})  
    df_prob.to_csv('data/stacking/prob_lgbm_test.csv',index=False) 
    
    
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
    lgbm=trainClassifierLGBM(feature_all,label_all)
    del feature_all
    del label_all
    test_all=stest.drop(['sort','label','clickTime','conversionTime',
                         'residence','appCategory'],axis=1).values 
    del stest
    
    prob = lgbm.predict(test_all, num_iteration=lgbm.best_iteration)
    list_prob+=prob.tolist()
    
    print(len(list_prob))
    del test_all
    del prob
    del lgbm
    
    print('开始stacking2')
    strain=train1.append([train3,train4,train5])
    stest=train2 
    label_all=strain['label'].values
    feature_all=strain.drop(['sort','label','clickTime','conversionTime',
                             'residence','appCategory'],axis=1).values  
    del strain
    lgbm=trainClassifierLGBM(feature_all,label_all)
    del feature_all
    del label_all
    test_all=stest.drop(['sort','label','clickTime','conversionTime',
                         'residence','appCategory'],axis=1).values 
    del stest
    
    prob = lgbm.predict(test_all, num_iteration=lgbm.best_iteration)
    list_prob+=prob.tolist()
    
    print(len(list_prob))
    del test_all
    del prob
    del lgbm
    
    print('开始stacking3')
    strain=train1.append([train2,train4,train5])
    stest=train3 
    label_all=strain['label'].values
    feature_all=strain.drop(['sort','label','clickTime','conversionTime',
                             'residence','appCategory'],axis=1).values  
    del strain
    lgbm=trainClassifierLGBM(feature_all,label_all)
    del feature_all
    del label_all
    test_all=stest.drop(['sort','label','clickTime','conversionTime',
                         'residence','appCategory'],axis=1).values 
    del stest
    
    prob = lgbm.predict(test_all, num_iteration=lgbm.best_iteration)
    list_prob+=prob.tolist()
    
    print(len(list_prob))
    del test_all
    del prob
    del lgbm
    
    print('开始stacking4')
    strain=train1.append([train2,train3,train5])
    stest=train4
    label_all=strain['label'].values
    feature_all=strain.drop(['sort','label','clickTime','conversionTime',
                             'residence','appCategory'],axis=1).values  
    del strain
    lgbm=trainClassifierLGBM(feature_all,label_all)
    del feature_all
    del label_all
    test_all=stest.drop(['sort','label','clickTime','conversionTime',
                         'residence','appCategory'],axis=1).values 
    del stest
    
    prob = lgbm.predict(test_all, num_iteration=lgbm.best_iteration)
    list_prob+=prob.tolist()
    
    print(len(list_prob))
    del test_all
    del prob
    del lgbm
    
    print('开始stacking5')
    strain=train1.append([train2,train3,train4])
    stest=train5 
    label_all=strain['label'].values
    feature_all=strain.drop(['sort','label','clickTime','conversionTime',
                             'residence','appCategory'],axis=1).values  
    del strain
    lgbm=trainClassifierLGBM(feature_all,label_all)
    del feature_all
    del label_all
    test_all=stest.drop(['sort','label','clickTime','conversionTime',
                         'residence','appCategory'],axis=1).values 
    del stest
    
    prob = lgbm.predict(test_all, num_iteration=lgbm.best_iteration)
    list_prob+=prob.tolist()
    
    print(len(list_prob))
    del test_all
    del prob
    del lgbm
    
    df_prob=pd.DataFrame({'sort':list_sort,'lgbm_prob':list_prob})
    
    df_prob=df_prob.sort(columns='sort')
    
    del train1
    del train2
    del train3
    del train4
    del train5
    
    df_prob.to_csv('data/stacking/prob_lgbm_train.csv',index=False) 

if __name__=='__main__':
    stacking_train()
    stacking_test()