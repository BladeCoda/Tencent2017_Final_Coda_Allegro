#-------用于挖掘关联特征-------
import numpy as np
import pandas as pd
    
#给年龄分段
def ageArea(x):
    if x==0:
        return 0
    elif x>=80:
        return 9
    else:
        return (x//10)+1
    
#统计appDetail
#V2-V3版特征更新
def actionDetail(x):
    
    recent_action=0#最近安装的APP数量
    recent_cat=0#最近安装同类APP数量
    
    c_time=int(str(x.clickTime))      
    
    if not isinstance(x.actionList,float):
        time_list=[int(i) for i in x.timeList.strip().split(' ')]
        action_list=x.actionList.strip().split(' ')
        type_list=x.action_type.strip().split(' ')
        
        for i in range(len(action_list)):
            if time_list[i]<c_time:
                
                recent_action+=1

                atype=int(type_list[i])
                
                if atype<100:
                    main=atype
                    sub=0
                else:
                    main=atype//100
                    sub=atype%100
                
                if x.appTypeMain==main and x.appTypeSub==sub:
                    recent_cat+=1
            else:
                #因为是按照升续排列，所以这里要可以跳出了
                break
    
    re=str(recent_action)+','+str(recent_cat)
    
    print(re)
    
    return re
    
def getrecent_action(x):
    return x.App_Detail.split(',')[0]

def getrecent_cat(x):
    return x.App_Detail.split(',')[1]

def digActionDetail():
    df_output=pd.read_csv('data/train30_v2.csv')
    save_path='data/train30_v3.csv'
    
    df_detail=pd.read_csv('../data/feature/action_list_v1.csv')
    
    df_output=pd.merge(df_output,df_detail,how='left',on='userID')
    
    print('拼接,开始统计')
    
    del df_detail
        
    df_output['App_Detail']=df_output.apply(lambda x:actionDetail(x),axis=1)
    
    print('APP细节挖掘完成')
    
    df_output['recent_action']=df_output.apply(lambda x:getrecent_action(x),axis=1)
    df_output['recent_cat']=df_output.apply(lambda x:getrecent_cat(x),axis=1)
    
    df_output['ageArea']=df_output['age'].apply(lambda x:ageArea(x))
    
    df_output=df_output.drop(['App_Detail','actionList','action_type','timeList'],axis=1)
    
    print('切割完成，开始保存')
    
    df_output.to_csv(save_path,index=False) 
    
#V5-V6版特征，加入了安装信息
def InstallDetail(x):
    
    is_Have=0#是否安装过本APP
    install_cat=0#安装同类APP数量
    install_num=0#安装APP的总数量    

    if not isinstance(x.appList,float):
        install_list=x.appList.strip().split(' ')
        type_list=x.install_type.strip().split(' ')
        
        for i in range(len(install_list)):
            install_num+=1
            if(install_list[i]==str(x.appID)):
                is_Have=1        
            
            atype=int(type_list[i])
            
            if atype<100:
                main=atype
                sub=0
            else:
                main=atype//100
                sub=atype%100
            
            if x.appTypeMain==main and x.appTypeSub==sub:
                install_cat+=1       
                
    re=str(is_Have)+','+str(install_num)+','+str(install_cat)
    print(re)   
    return re
    
def get_is_Have(x):
    return x.Install_Detail.split(',')[0]

def get_install_num(x):
    return x.Install_Detail.split(',')[1]

def get_install_cat(x):
    return x.Install_Detail.split(',')[2]

def digInstallDetail():
    in_path='data/train30_v5.csv'
    save_path='data/train30_v6.csv'
    df_all=pd.read_csv('data/train30_v1.csv')
    
    df_detail=pd.read_csv('../data/feature/installed_list_v1.csv')
    df_all=pd.merge(df_all,df_detail,how='left',on='userID')
    
    print('拼接,开始统计')
    
    del df_detail
        
    df_all['Install_Detail']=df_all.apply(lambda x:InstallDetail(x),axis=1)
    
    print('APP细节挖掘完成')
    
    df_all['is_Have']=df_all.apply(lambda x:get_is_Have(x),axis=1)
    print('is_Have统计完成')
    df_all['install_num']=df_all.apply(lambda x:get_install_num(x),axis=1)
    print('install_num统计完成')
    df_all['install_cat']=df_all.apply(lambda x:get_install_cat(x),axis=1)
    print('install_cat统计完成')
    
    isHave=df_all['is_Have'].values
    installNum=df_all['install_num'].values
    installCat=df_all['install_cat'].values

    del df_all
    
    print('开始合并') 
    df_output=pd.read_csv(in_path)
    df_output['isHave']=isHave
    df_output['install_num']=installNum
    df_output['install_cat']=installCat

    print('开始保存')
    
    df_output.to_csv(save_path,index=False)

    
if __name__=='__main__':
    #digActionDetail()  
    digInstallDetail()