# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import xgboost as xgb

#XGB的模型训练代码

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
def trainClassifierXGB(x_train,y_train,f_names=None,buffer=False):
    #用30W条数据测试下（只要是调参）
    
    print('使用XGBOOST进行训练')
    if buffer==True:
        dtrain=xgb.DMatrix("data/train.buffer")
    else:
        dtrain=xgb.DMatrix(x_train,label=y_train,feature_names=f_names)
        #保存缓存
        #dtrain.save_binary("data/train.buffer")
    #设置缺省值。这儿不用
    #dtrain = xgb.DMatrix( data, label=label, missing=0)
    
    #参数设置
    #eta[默认0.3]：通过减少每一步的权重，可以提高模型的鲁棒性。 典型值为0.01-0.2
    #min_child_weight[默认1]：XGBoost的这个参数是最小样本权重的和，而GBM参数是最小样本总数。 这个参数用于避免过拟合。当它的值较大时，可以避免模型学习到局部的特殊样本。 但是如果这个值过高，会导致欠拟合。这个参数需要使用CV来调整。
    #max_depth[默认6]:和GBM中的参数相同，这个值为树的最大深度。 这个值也是用来避免过拟合的。max_depth越大，模型会学到更具体更局部的样本
    #subsample[默认1]和GBM中的subsample参数一模一样。这个参数控制对于每棵树，随机采样的比例。 减小这个参数的值，算法会更加保守，避免过拟合。但是，如果这个值设置得过小，它可能会导致欠拟合。 典型值：0.5-1
    #colsample_bytree:和GBM里面的max_features参数类似。用来控制每棵随机采样的列数的占比(每一列是一个特征)。 典型值：0.5-1
    #(没咋用)gamma [default=0]：模型在默认情况下，对于一个节点的划分只有在其loss function 得到结果大于0的情况下才进行，而gamma 给定了所需的最低loss function的值gamma值使得算法更conservation，且其值依赖于loss function ，在模型中应该进行调参。
    #scale_pos_weight[default=1]A value greater than 0 can be used in case of high class imbalance as it helps in faster convergence.大于0的取值可以处理类别不平衡的情况。帮助模型更快收敛
    #binary:logistic返回预测的概率(不是类别,正类)
    #lambda[默认1]权重的L2正则化项。(和Ridge regression类似)。 这个参数是用来控制XGBoost的正则化部分的。
    #--------笔记--------
    #基本只要改max_depth,eta,num_round这3个，其他的没发现什么太大的提高
    #学习率（eta）调小,num_round调大可以提高效果，但是如果eta太大，num_round也大效果会很差

    #0.25 200
    #0.1 500
    
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
    
#encoding代表是不是有独热编码信息（需要分解）
def predict_test_prob(bst):
    df_all=loadCSV('data/first_merge/test_join_v9.csv') 
    
    df_sta_lgbm=loadCSV('data/stacking/prob_lgbm_test.csv') 
    print('开始拼接')
    df_all=pd.merge(df_all,df_sta_lgbm,how='left',on='instanceID')
    del df_sta_lgbm
    
    instanceID=df_all.instanceID.values
    feature_all=df_all.drop(['label','clickTime','instanceID',
                             'residence','appCategory'],axis=1).values
                             
    del df_all
                             
    dtest=xgb.DMatrix(feature_all)
    prob=bst.predict(dtest)
    
    output=pd.DataFrame({'instanceID':instanceID,'prob':prob})
    
    output.to_csv('result/submission2.csv',index=False) 

#交叉验证
def cross_validat(df_all,test_size=0.2):
    
    feature_all=df_all.drop(['sort','label','clickTime','conversionTime',
                             'residence','appCategory'],axis=1).values
                             
    fnames=df_all.drop(['sort','label','clickTime','conversionTime',
                        'residence','appCategory'],axis=1).columns
        
    label_all=df_all.label.values
    
    del df_all

    x_train,x_test,y_train,y_test=train_test_split(feature_all,label_all,test_size=test_size,random_state=42)
    print('数据集切割完成')
    
    del feature_all
    del label_all
    
    bst=trainClassifierXGB(x_train,y_train,f_names=fnames)
    print('训练完成')
    #输出特征的importance
    print(bst.get_score())
    
    dtest=xgb.DMatrix(x_test,feature_names=fnames)
    
    prob=bst.predict(dtest)
    
    loss=logloss(y_test,prob)
    print('交叉验证损失为:',loss)
    
    return prob
    
    
#主函数入口
if __name__=='__main__':
   df_all=loadCSV('data/cutData/train_time_v9.csv') 
   print('CSV加载完成')
   
   df_2=loadCSV('day30/data/train30_v9.csv') 
   print('开始拼接')
   df_all=df_all.append(df_2)
   del df_2
   
   #原始:0.09321
   
   df_sta_lgbm=loadCSV('data/stacking/prob_lgbm_train.csv') 
   print('开始拼接')
   df_all=pd.merge(df_all,df_sta_lgbm,how='left',on='sort')
   del df_sta_lgbm
   
   #pre=cross_validat(df_all,test_size=0.2)
   
   feature_all=df_all.drop(['sort','label','clickTime','conversionTime',
                            'residence','appCategory'],axis=1).values                         
                            
   label_all=df_all.label.values
   
   del df_all
   
   bst=trainClassifierXGB(feature_all,label_all)
   print('分类器训练完成')
   predict_test_prob(bst)
   print('结果预测完成')
   