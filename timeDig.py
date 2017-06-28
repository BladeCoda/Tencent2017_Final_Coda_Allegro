# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

#基于时序的一些挖掘代码

#V4-V5特征

def digTimeSpam(x):

    timeSpam=-1#保存与上条点击记录的时间间隔,-1代表没有间隔
    timeSpam_app=-1#保存与上条点击记录的时间间隔(同app),-1代表没有间隔
    
    if not isinstance(x.cl_time,float):
        time_list=[int(i) for i in x.cl_time.strip().split(' ')]
        app_list=[int(i) for i in x.cl_app.strip().split(' ')]
                  
        lasttime=-1
        lasttime_app=-1
        
        for i in range(len(time_list)):
            #跳出
            if(time_list[i]>x.clickTime):
                break
            elif time_list[i]==x.clickTime:
                #当前是重复数据
                if x.isRepeat>0:
                    timeSpam=0
                    if x.appID==app_list[i]:
                        timeSpam_app=0
                break
            else:
                lasttime=time_list[i]
                if x.appID==app_list[i]:
                     lasttime_app=time_list[i]

        if timeSpam!=0:
            if lasttime==-1:
                timeSpam=-1
            else:
                timeSpam=x.clickTime-lasttime
            
        if timeSpam_app!=0:
            if lasttime_app==-1:
                timeSpam_app=-1
            else:
                timeSpam_app=x.clickTime-lasttime_app    
                
    ct_usr=dic_count['ct'+str(x.clickTime//100)+','+'usr'+str(x.userID)]
    ct_app=dic_count['ct'+str(x.clickTime//100)+','+'app'+str(x.appID)]
    ct_cre=dic_count['ct'+str(x.clickTime//100)+','+'cre'+str(x.creativeID)]
    
    re=str(timeSpam)+','+str(timeSpam_app)
    re+=','+str(ct_usr)+','+str(ct_app)+','+str(ct_cre)
    
    print(re)
    return re

def get_timeSpam(x):
    return x.split(',')[0]

def get_timeSpam_app(x):
    return x.split(',')[1]

def get_ct_usr(x):
    return x.split(',')[2]

def get_ct_app(x):
    return x.split(',')[3]

def get_ct_cre(x):
    return x.split(',')[4]


def repeatFeature(data_type='train'):
    inpath=''
    outpath=''
    if data_type=='train':
        inpath='data/cutData/train_time_v4.csv'
        outpath='data/cutData/train_time_v5.csv'
        df_all=pd.read_csv('data/origin/train.csv').drop(['label','conversionTime'],axis=1) 
    elif data_type=='test':
        inpath='data/first_merge/test_join_v4.csv'
        outpath='data/first_merge/test_join_v5.csv'
        df_all=pd.read_csv('data/origin/test.csv').drop(['label','instanceID'],axis=1) 
    else:
        print('data_type出错！')
        return
           
    df_all['clickTime']=df_all.clickTime.apply(lambda x:x//100)    
    
    feature_all=df_all.values.tolist()
    
    repeatList=[]
    total=len(feature_all)
    
    print('csv加载完成，开始计算') 
    flag=0
    for i in range(total):
        if flag==0:
            if i+1<total and feature_all[i]==feature_all[i+1]:
                flag=1
            repeatList.append(0)
        else:           
            repeatList.append(flag)
            flag+=1
            if i+1<total and feature_all[i]!=feature_all[i+1]:
                flag=0
            
        if i%100000==0:
            print('已统计 %.2f %%'%(i/total*100))
        
    df_repeat=pd.DataFrame({'sort':[i for i in range(len(df_all))],'isRepeat':repeatList})
        
    del df_all
    del feature_all    
    
    if data_type=='train':
        df_all2=pd.read_csv(inpath)
        df_all2=pd.merge(df_all2,df_repeat,how='left',on='sort')
    else:
        df_all2=pd.read_csv(inpath)
        df_all2['isRepeat']=repeatList

    del repeatList
    del df_repeat

    df_detail=pd.read_csv('data/feature/click_detail_v1.csv')
    
    df_all2=pd.merge(df_all2,df_detail,how='left',on='userID')
    print('开始统计')
        
    df_all2['timeDetail']=df_all2.apply(lambda x:digTimeSpam(x),axis=1)
    
    print('点击同time_m次数统计完成')
    df_all2['timeSpam']=df_all2['timeDetail'].apply(lambda x:get_timeSpam(x))
    print('点击同timeSpam次数统计完成')
    df_all2['timeSpam_app']=df_all2['timeDetail'].apply(lambda x:get_timeSpam_app(x))
    print('点击同timeSpam_app次数统计完成')
    
    df_all2['ct_usr']=df_all2['timeDetail'].apply(lambda x:get_ct_usr(x))
    print('ct_usr统计完成')
    df_all2['ct_app']=df_all2['timeDetail'].apply(lambda x:get_ct_app(x))
    print('ct_app统计完成')
    df_all2['ct_cre']=df_all2['timeDetail'].apply(lambda x:get_ct_cre(x))
    print('ct_cre统计完成')
    
    print('time细节挖掘完成')
    df_all2=df_all2.drop(['timeDetail','cl_time','cl_app','cl_label'],axis=1)

    df_all2.to_csv(outpath,index=False)
    

#构造组合标签字典
def createTimeDict():

    df_train=pd.read_csv('data/cutData/train_time_v1.csv').drop(['conversionTime','sort'],axis=1)
    df_test=pd.read_csv('data/first_merge/test_join_v1.csv').drop(['instanceID'],axis=1)
    
    df_all=pd.concat([df_train,df_test])
    df_all['clickTime']=df_all.clickTime.apply(lambda x:x//100)
    
    del df_train
    del df_test
    
    dic_count={}
    
    #ct+usr
    temp=df_all.groupby(['clickTime','userID'])
    for name,df_t in temp:
        iname='ct'+str(name[0])+','+'usr'+str(name[1])
        dic_count[iname]=len(df_t)
    print('ct+usr构造完成')
    
    #ct+app
    temp=df_all.groupby(['clickTime','appID'])
    for name,df_t in temp:
        iname='ct'+str(name[0])+','+'app'+str(name[1])
        dic_count[iname]=len(df_t)
    print('ct+app构造完成')
    
    #ct+cre
    temp=df_all.groupby(['clickTime','creativeID'])
    for name,df_t in temp:
        iname='ct'+str(name[0])+','+'cre'+str(name[1])
        dic_count[iname]=len(df_t)
    print('ct+cre构造完成')
            
    print('字典构造完成')
    
    del df_all
    return dic_count
       


#-------v6-v7特征------------
def digTimeSpamRe(x):

    timeSpam_re=-1
    timeSpam_app_re=-1
    
    if not isinstance(x.cl_time,float):
        time_list=[int(i) for i in x.cl_time.strip().split(' ')]
        app_list=[int(i) for i in x.cl_app.strip().split(' ')]
                  
        lasttime_re=-1
        lasttime_app_re=-1
        
        time_list.reverse()
        app_list.reverse()
        
        for i in range(len(time_list)):
            #跳出
            if(time_list[i]<x.clickTime):
                break
            elif time_list[i]==x.clickTime:
                #当前是重复数据
                if x.isRepeat_re>0:
                    timeSpam_re=0
                    if x.appID==app_list[i]:
                        timeSpam_app_re=0
                break
            else:
                lasttime_re=time_list[i]
                if x.appID==app_list[i]:
                     lasttime_app_re=time_list[i]

        if timeSpam_re!=0:
            if lasttime_re==-1:
                timeSpam_re=-1
            else:
                timeSpam_re=lasttime_re-x.clickTime
            
        if timeSpam_app_re!=0:
            if lasttime_app_re==-1:
                timeSpam_app_re=-1
            else:
                timeSpam_app_re=lasttime_app_re-x.clickTime
    
    re=str(timeSpam_re)+','+str(timeSpam_app_re)
    
    print(re)
    return re
    
def get_timeSpam_Re(x):
    return x.split(',')[0]

def get_timeSpam_app_Re(x):
    return x.split(',')[1]

#反向统计间隔
def repeatFeatureRe(data_type='train'):
    inpath=''
    outpath=''
    if data_type=='train':
        inpath='data/cutData/train_time_v6.csv'
        outpath='data/cutData/train_time_v7.csv'
        df_all=pd.read_csv('data/origin/train.csv').drop(['label','conversionTime'],axis=1) 
    elif data_type=='test':
        inpath='data/first_merge/test_join_v6.csv'
        outpath='data/first_merge/test_join_v7.csv'
        df_all=pd.read_csv('data/origin/test.csv').drop(['label','instanceID'],axis=1) 
    else:
        print('data_type出错！')
        return
           
    df_all['clickTime']=df_all.clickTime.apply(lambda x:x//100)    
    
    feature_all=df_all.values.tolist()
    repeatList_re=[]
    total=len(feature_all)  
    feature_all.reverse()  
    
    flag=0
    for i in range(total):
        if flag==0:
            if i+1<total and feature_all[i]==feature_all[i+1]:
                flag=1
            repeatList_re.append(0)
        else:          
            repeatList_re.append(flag)
            flag+=1
            if i+1<total and feature_all[i]!=feature_all[i+1]:
                flag=0
            
        if i%100000==0:
            print('已统计 %.2f %%'%(i/total*100))
    
    repeatList_re.reverse()
    print('拼接完成，开始读取csv')
    
    df_repeat=pd.DataFrame({'sort':[i for i in range(len(df_all))],'isRepeat_re':repeatList_re})
        
    del df_all
    del feature_all    
    
    if data_type=='train':
        df_all2=pd.read_csv('data/cutData/train_time_v1.csv')
        df_all2=pd.merge(df_all2,df_repeat,how='left',on='sort')
    else:
        df_all2=pd.read_csv('data/first_merge/test_join_v1.csv')
        df_all2['isRepeat_re']=repeatList_re

    del repeatList_re
    del df_repeat

    df_detail=pd.read_csv('data/feature/click_detail_v1.csv')
    
    df_all2=pd.merge(df_all2,df_detail,how='left',on='userID')
    print('开始统计')
    
    del df_detail
        
    df_all2['timeDetail']=df_all2.apply(lambda x:digTimeSpamRe(x),axis=1)
    
    df_all2['timeSpam_re']=df_all2['timeDetail'].apply(lambda x:get_timeSpam_Re(x))
    print('点击同timeSpam次数统计完成')
    df_all2['timeSpam_app_re']=df_all2['timeDetail'].apply(lambda x:get_timeSpam_app_Re(x))
    print('点击同timeSpam_app次数统计完成')
    
    print('导出数据')
    
    isRepeat_re=df_all2.isRepeat_re.values
    timeSpam_re=df_all2.timeSpam_re.values
    timeSpam_app_re=df_all2.timeSpam_app_re.values
    
    del df_all2
    
    df_all2=pd.read_csv(inpath)
    df_all2['isRepeat_re']=isRepeat_re
    df_all2['timeSpam_re']=timeSpam_re
    df_all2['timeSpam_app_re']=timeSpam_app_re

    print('开始保存')
    df_all2.to_csv(outpath,index=False)
    
dic_count=createTimeDict()

if __name__=='__main__':
    #repeatFeature(data_type='train')
    #repeatFeature(data_type='test')
    
    repeatFeatureRe(data_type='train')
    repeatFeatureRe(data_type='test')