#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
import csv
import math
import os
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import time
import pickle 
import random 
import torch 
import warnings
warnings.filterwarnings('ignore') 
    
'********* import surrogates *********'
#from GP import gp
#from GP2 import GPy
#from MARS import mars 
#from RBF import RBF_setoption,RBF_Build,RBF_predict

'********* import sampling functions *********'
from distance import distance
from Pareto import pareto 
from EEPA import EEPAfunc

'********* import test problems *********'
from TestProblems import * #rosenbrock,rastrigin,levy,Michalewicz,Perm,Zakharov,evaluate, GriewankRosenBrock, Rastriginn, Schaffers10

'********* import pool generators *********'
from Generator import generator 

'******* import other models *******'
from Models import RF, Lasso, PLS, calculate_vips, Cal_shap
from Other import func_reader

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

def gp(XX,YY,d):
    #ker= RBF()
    #Fit an GP model
    #ker = RBF()
    #RBF(length_scale=1.0)
    ker=ConstantKernel(1, (0.01, 100)) * RBF(length_scale=0.5 * np.ones(d,), length_scale_bounds=(0.05, 2.0)) +WhiteKernel(1e-4,(1e-6, 1e-2))
    #ker = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

    model = GaussianProcessRegressor(kernel=ker,random_state=0,n_restarts_optimizer=5)
    model.fit(XX.to_numpy(),YY.to_numpy())
    
    #y_hat = model.predict(XX_ran.to_numpy())

    return model


def R_rule(imp,r,cnimp,cimp):
    
    if(1<=cimp<=2):
        r=r
    elif(cimp>2):
        r=r*2
    elif(cnimp>=1 and r>=0.1):  #0.01 and 0.05 
        r=r/2
    elif(cnimp>1 and r<0.1):
        r=0.5 #0.2 and 0.5 
        
    return r 

#Sphere1,Ellipsoid2,Rastrigin3,BuecheRastrigin4, LinearSlope5
#AttractiveSector6, StepEllipsoid7, Rosenbrock8, RosenbrockRotated9
#EllipsoidRotated10, Discus11, BentCigar12, SharpRidge13, DifferentPowers14
#RastriginRotated15, Weierstrass16, Schaffers17, Schaffers18, GriewankRosenBrock19
#Schwefel20, Gallagher21, Gallagher22, Katsuura23, LunacekBiRastrigin24

def scaled_array(arr):
    arr_min = arr.min()
    arr_max = arr.max()
    scaled_arr = (arr - arr_min) / (arr_max - arr_min)
    return scaled_arr

def Main(budget,func,d,pool_gen,num_cand,k,mode,sampling,wd,poolname):

    ## reading cvs files from the folder pools 
    path,lb,up=func_reader(func,d,wd,poolname) 
    
    ## read the initial labeled data 
    file = pd.read_csv(path, index_col=0, header=0).round(10) 
    ## initializations 
    n1=evaluate(file,func)
    minpt=[]
    Yfitted=[]
    s=2
    iteration=1 
    rep=0 
    minY=n1.iloc[:,d+s-1].min() #value of min y
    pt=n1.iloc[:,d+s-1].idxmin() #index of min y
    nofe=(rep+1)*n1.shape[0] #budget (31)
    #minpt=minpt.append(n1.iloc[pt,0:d+s].append(pd.Series([nofe,0,0,0]),ignore_index=True)) # 0:33  
    p=0
    newsol=0
    r=2 #0.2*(up-lb)
    imp=0
    EEPAdist=1
    cimp=0
    cnimp=0
    initial_size=2*(d+1)
    yhat_vals=[]
    dist_vals=[]
    acq_values=[]
    bks_vals=[]
    bks=minY
    yhat_vals.append(np.zeros(initial_size))
    dist_vals.append(np.zeros(initial_size))
    acq_values.append(np.zeros(initial_size))
    bks_vals.append(np.ones(initial_size)*bks)
    srg_imp=[]
    while nofe<=budget: 
        
        print('Iteration number: ',iteration)
        ## define the initial labeled data 
        xx = n1.iloc[:,0:d]
        yy = n1.iloc[:,d+s-1]
        
        bestf=n1.iloc[:,d+s-1].min()
        bks_vals.append(np.array([bestf]*k))
        print("best f",bestf)
        print(yy)
        #print('n1 is:',n1)
        #print('Min y:',min(yy))
        ## check the R rule and update r 
        r=R_rule(imp,r,cnimp,cimp)
        if(r>=2): 
            r=2
        print('This r:',r)
        
        model=gp(xx,yy,d)
        #from pyearth import Earth
        #int=np.int
#         np.int = int
#         np.float=float 
    
        
# #         from pyearth import Earth
# #         model = Earth(max_degree=2, use_fast=True, fast_K=12, feature_importance_type='rss')
# #         model.fit(xx,yy)
# #         importances = model.feature_importances_
# #         srg_imp.append(importances)
#         #print('srg feature importance:',srg_imp)
#         #print('Aq values',acq_values)
#         #print('yhat values',yhat_vals)

        
#         del np.int
#         del np.float
        
        
        from sklearn.tree import DecisionTreeRegressor
#         model = DecisionTreeRegressor()
#         model.fit(xx, yy)
#         importances = model.feature_importances_
#         srg_imp.append(importances)
        
        '*** Setting Pool Generation Strategy ***'
        
        print('nofe is:',nofe)
        trg=0
        vv=k-trg
        
        print('Modified dynamic pool with RF F_imp')
        rf,F_imp=RF(xx,yy)
        prob_perturb=F_imp
        
        while (vv>0):  
            
            #print('Modified dynamic pool with RF F_imp')
            #rf,F_imp=RF(xx,yy)
            #prob_perturb=F_imp
            #print('Prob_perturb/F_imp:',prob_perturb)    
            ## dynamic generation part 
            from scipy.stats import truncnorm

            xbest=n1.sort_values(by=[d]).head(1).iloc[:,0:d]
            subset = np.arange(0, d)
            scalefactors = r

            # Generate candidate points
            if len(subset) == 1:  # Fix when nlen is 1
                ar = np.ones((num_cand, 1))
            else:
                ar = np.random.rand(num_cand, len(subset)) < prob_perturb
                ind = np.where(np.sum(ar, axis=1) == 0)[0]
                ar[ind, np.random.randint(0, len(subset) - 1, size=len(ind))] = 1
            cand = np.multiply(np.ones((num_cand, d)), np.array(xbest))
            for i in subset:
                lower, upper, sigma = lb, up, scalefactors
                ind = np.where(ar[:, i] == 1)[0]
                p=truncnorm.rvs(
                    a=(lower - xbest.iloc[:,i]) / sigma, b=(upper - xbest.iloc[:,i]) / sigma, loc=xbest.iloc[:,i], scale=sigma, 
                    size=len(ind))
                cand[ind, subset[i]] =p  
            sp=pd.DataFrame(cand)
            
            
            sp=pd.DataFrame(np.random.rand(num_cand,d))
            
            XX=xx.iloc[:,0:d] # x value without leave nodes
            YY=yy.copy() # copy yy value
            new=evaluate(sp,func)
            XX_ran = new.iloc[:,0:d]
            
            Yfit_random=model.predict(XX_ran.to_numpy())
            print('GPs Prediction range: ',Yfit_random.min(),Yfit_random.max())
            #print('MARS Prediction range: ',Yfit_random.min(),Yfit_random.max())
            
            #print('****EEPA Original Strategy****')
            p2,nn1=EEPAfunc(XX,XX_ran,Yfit_random,func,d,k,EEPAdist)
            #print('nn1',nn1)
            
            nn1=pd.DataFrame(nn1[0])
            
            #print('yhat values',yhat_vals)
            n=nn1.iloc[0:vv,list(range(d))]
            #SS=n.copy()
            
            nn2=nn1.iloc[0:vv,:]
            Yhat_selected=nn2.iloc[:,d].drop_duplicates() #yhat
            dist_selected=nn2.iloc[:,d+1].drop_duplicates() #dist 
            yhat_vals.append(Yhat_selected.to_numpy())
            dist_vals.append(dist_selected.to_numpy())
            inst_aq=dist_selected.to_numpy()-Yhat_selected.to_numpy()
            print('inst_aq',inst_aq,dist_selected.to_numpy(),Yhat_selected.to_numpy())
            if(inst_aq.shape[0]>1):
                scaled_aq=scaled_array(inst_aq)
            else:
                scaled_aq=np.array([1])
            acq_values.append(scaled_aq)
            print('Aq values',acq_values)
            
            n=pd.concat([n, nn1.loc[:,'index']],axis=1, ignore_index=True) 
            n=n.drop_duplicates()
            n=n.dropna()
            SS=n.copy()
            #print('SS',SS)
            
            #Yhat_selected=nnmodel.predict(SS.iloc[:,0:d].to_numpy())
            #print(Yhat_selected)
            #print('yhat,dist',Yhat_selected,dist_selected)
            
            vv=vv-SS.shape[0]
            #print('vv',vv)
            
            idss=np.array(n.iloc[:,d].T.astype('int'))
            #print(idss)
            y=new.iloc[idss,d:d+1] #d+s+rep
            print('new ys: ',y)
            arrr=new.iloc[n.iloc[:,d],:].reset_index(drop=True)
            n1=n1.reset_index(drop=True)
            arrr.columns=n1.columns
            #print('this part',arrr)
            #print('n1 before',n1)
            n1=pd.concat([n1,arrr],axis=0,ignore_index=True).reset_index(drop=True)
            #print('n1 here is ',n1)
            minY=n1.iloc[:,d+s-1].min()
            pt=n1.iloc[:,d+s-1].idxmin()
            nofe=n1.shape[0] #minpt[iteration-1][d+2]+(rep+1)*n.shape[0]

        print('nofe is :' ,nofe)
        
        #if(sampling==0):
            #minpt.append(n1.iloc[pt,0:d+s].append(pd.Series([nofe,p2.shape[0],p]),ignore_index=True)) 
        #else:
            #minpt.append(n1.iloc[pt,0:d+s].append(pd.Series([nofe,XX_ran.shape[0],p]),ignore_index=True))
            
        sol=n1.iloc[:,d+s-1].min()
        #bks_vals.append(np.array(sol)*k)
        #print(bks_vals)
        #print(yhat_vals)
        thre=10**math.floor(math.log10(abs(sol)))
        thre=thre/2 
        imp=newsol-sol
        
        if(imp>thre): 
            cimp=cimp+1
            cnimp=0
        if(imp<thre): 
            cnimp=cnimp+1
            cimp=0
        print('Threshold:',thre)
        print('Val,cImp,cNotimp',imp,cimp,cnimp)
        
        newsol=sol    
        inds=n1.iloc[:,d+s-1].idxmin()
        best_point=pd.DataFrame([inds,sol])
        print('Current Solution: ',sol)
        iteration=iteration+1
        
        print('******************************************')
    
    return n1,yhat_vals,dist_vals,acq_values,bks_vals,srg_imp #minpt,d,iteration,nofe,sol


def read_run(budget,func,d,pool_gen,num_cand,k,mode,sampling):

    start_time = time.time()
    pg=pool_gen
    ss=sampling 
    import os
    wd=os.getcwd()
    times=[]
    
    Name='Levy_6d' #'Gallagher_30d' #change this 
    for i in range(1,6,1): 
        print(i)    
        a='**** Running round No Replication'+str(i)+'_Strategy_'+str(pg)+str(ss)+'****'
        print(a)
        poolname='pool_'+str(i) #+'.csv'
        rndpoolname='new_'+str(i) #+'.csv'
        out_arr=np.array([i])

        #minpt,d,iteration,nofe,sol=Main(budget,func,d,pool_gen,num_cand,k,mode,sampling,wd,poolname)
        n1,yhat_vals,dist_vals,acq_values,bks_vals,srg_imp=Main(budget,func,d,pool_gen,num_cand,k,mode,sampling,wd,poolname)
        t=(time.time() - start_time)
        times.append(t)
        print('Computation time for',a,'is: ',t)
        
        folder_name = 'result_'+Name
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        else:
            print(f"The folder '{folder_name}' already exists.")
        Name='Levy'
        df = n1.iloc[:,0:d] #pd.DataFrame(train_X.numpy(), columns=np.arange(1,d+1).tolist())
        df['f(x)'] = n1.iloc[:,d] #train_Y.numpy()
        all_acquisition_values = np.hstack(acq_values).reshape(1, df.shape[0])[0]
        df['acquisition_value'] = all_acquisition_values
        df['mean']= np.hstack(yhat_vals).reshape(1, df.shape[0])[0]
        df['distance']= np.hstack(dist_vals).reshape(1, df.shape[0])[0]
        df['BKS']= np.hstack(bks_vals).reshape(1, df.shape[0])[0]
        df.to_csv('Levy_6d_DEEPA_'+str(i)+'.csv',index=False) 
        
    return None # n1,yhat_vals,dist_vals,acq_values,bks_vals,srg_imp,i #df,srg_imp



read_run(budget=400,func=3,d=6,pool_gen=1,num_cand=600,k=8,mode=3,sampling=0)




# In[ ]:




