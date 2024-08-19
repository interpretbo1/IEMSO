#!/usr/bin/env python
# coding: utf-8

# In[3]:


import math
import os 
import pandas as pd 
import numpy as np
import time 
import pickle 

import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.utils.transforms import standardize, normalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler #.samplers
import gpytorch
import math 

from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy
## import acquistion functions 

from botorch.acquisition import qKnowledgeGradient,qExpectedImprovement,qProbabilityOfImprovement
from botorch.acquisition import qUpperConfidenceBound,qMaxValueEntropy
from botorch.utils import draw_sobol_samples


## define the functions here 
def rosenbrock(x):
    n=len(x)
    return -sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(n - 1))

def michalewicz(X):
    m=10
    i = np.arange(1, X.shape[-1] + 1)
    term1 = np.sin(X)
    term2 = np.sin(i * X**2 / np.pi)**(2 * m)
    return -np.sum(term1 * term2, axis=-1)

def rastrigin(X):
    A = 10
    n = X.shape[-1]
    return - (A * n + np.sum(X**2 - A * np.cos(2 * np.pi * X), axis=-1))

# def levy(x):
#     # Convert x to numpy array to ensure we can handle array operations
#     #x = np.array(x)
#     #print(x,w)
#     x=x.T
#     w = 1 + (x - 1) / 4
#     #print(x,w)
#     term1 = np.sin(np.pi * w[0]) ** 2
#     term2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
#     term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
#     y = term1 + term2 + term3    
#     return -y

def levy(X):
    # X is expected to be a numpy array of shape (n, 6) where n is the number of rows (data points)
    w = 1 + (X - 1) / 4

    term1 = np.sin(np.pi * w[:, 0]) ** 2
    term2 = np.sum((w[:, :-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:, :-1] + 1) ** 2), axis=1)
    term3 = (w[:, -1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[:, -1]) ** 2)

    y = term1 + term2 + term3

    # Reshape the output to be (n, 1)
    return -y.reshape(-1, 1)

def bayopt(acq='EI',budget=50,pool_size=1000,batch_size=4,initial_size=14,d=6,pname='Rastrigin'):


    wd=os.getcwd()
    
    idxpool=np.arange(1,6) #10 pools 
    
    #Modify function name, folder name, bounds 
    func=levy #rastrigin #michalewicz #rosenbrock #branin #rosenbrock2 #six_hump_camel
    name='Levy_6d' #'Rast_10d' #'Mich_10d' #Rosen_10d
    lb= -5 #-5.12 #0 #-2.048 #[-5.12, 5.12]
    up= 5 #5.12 #np.pi #2.048
    b=torch.ones([2,d])
    b[0]=b[0]*lb
    b[1]=b[1]*up
    bounds=b
    bounds = bounds.double()
    
    #print('bounds:',bounds)
        
    print('*** acquisition function is: ',acq,'***')    
    for i in idxpool:
        start=time.time()
        B=budget-initial_size

        poolname='pool_'+str(i) #+'.csv'
        print('****Initialization: ',poolname,' ****')
        path1 = wd+'/'+name+'/'+str(poolname) # 

        train_X=pd.read_csv(path1, index_col=0, header=0).round(3).iloc[:,0:d].to_numpy()
        print(train_X)
        train_Y = func(train_X).reshape(-1,1)
        print(train_Y)
        bks=-train_Y.max()
        print('bks',bks)
        train_X=torch.from_numpy(train_X)
        train_Y=torch.from_numpy(train_Y)
        #print(train_X,train_Y)
        yvals=[]
        t=0
        acq_values=[]
        var=[]
        yhat=[]
        acq_values.append(np.zeros(initial_size))
        yvals.append(np.ones(initial_size)*bks)
        var.append(np.zeros(initial_size))
        yhat.append(train_Y.numpy().reshape(1,-1)[0])
        while (0<B):
            from gpytorch.kernels import RBFKernel
            from gpytorch.likelihoods import GaussianLikelihood
            
            t=t+1
            print('Remained Budget',B)
            print('Labeled Set Size',train_X.shape)
            best_f = train_Y.max()
            print('bestf',-best_f)
            yvals.append(np.array([-best_f]*batch_size))
            
            # Define the GP model components
            
            # Define the GP model components
            likelihood = GaussianLikelihood() 
            likelihood.noise_prior=1e-4
            likelihood.noise_constraint=(1e-6, 1e-2)
            
            from gpytorch.kernels import MaternKernel
            from gpytorch.means import ConstantMean
            from gpytorch.kernels import RBFKernel, ScaleKernel
            
            # Define the kernel components         
            rbf_kernel = RBFKernel()
            rbf_kernel.lengthscale_prior=0.5 * np.ones(train_X.shape[-1])
            rbf_kernel.length_scale_constraint= (0.05, 2.0)
            covv = rbf_kernel

            meann=ConstantMean()
            meann.constant_prior=1
            meann.constant_constraint=(0.01, 100)
            
            #kernel = RBFKernel()  # Radial Basis Function (RBF) kernel for GP
            # Initialize the SingleTaskGP model
            model = SingleTaskGP(train_X, train_Y, likelihood=likelihood, mean_module=meann,  covar_module=covv)
            
            #model = SingleTaskGP(train_X, train_Y)
            
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            with gpytorch.settings.cholesky_jitter(1e-3):
                fit_gpytorch_model(mll)

            ## sobol is the better than iid 
            sampler=SobolQMCNormalSampler(pool_size)
            ## random candidate set for acq 
            #print(bounds.dtype,train_X.dtype, train_Y.dtype)
            candidate_set = torch.rand(pool_size, bounds.size(1), device=bounds.device, dtype=bounds.dtype)
            candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set
            #candidate_set = draw_sobol_samples(bounds=bounds, n=1000, q=1).squeeze(1)
            
            if(acq=='EI'):
                aq = qExpectedImprovement(model,best_f,sampler) 
                seq=True
            elif(acq=='PI'):
                aq = qProbabilityOfImprovement(model,best_f,sampler)
                seq=True
            elif(acq=='KG'):
                #sampler2=SobolQMCNormalSampler(64) #this is the bestone apparently 
                #n=64
                aq = qKnowledgeGradient(model)
                seq=True 
            elif(acq=='UCB'):
                beta=2*np.log((pool_size*t*t*9)/6)
                aq = qUpperConfidenceBound(model,beta,sampler)
                seq=True 
            elif(acq=='MES'):
                aq = qMaxValueEntropy(model, candidate_set) 
                seq=True 
            elif(acq=='Gibbon'):
                aq = qLowerBoundMaxValueEntropy(model, candidate_set)
                seq=True
            else:
                print('No Valid Acquisition Function Selected')
                break
                
            
            candidates, acq_value = optimize_acqf(
                acq_function=aq, 
                bounds=bounds,
                q=batch_size, #batch_size
                num_restarts=2,
                raw_samples=pool_size,
                sequential=seq) 

            #MES face problem with sequentional=False
            #sequential: If False, uses joint optimization, otherwise uses sequential optimization
            
            acq_values.append(acq_value.numpy())
            print(acq_value.numpy())
#             print(candidates)
#             for kk in range(batch_size):
#                 # Get the posterior distribution
            #print(candidates.dtype)
            posterior = model.posterior(candidates)
            mean = posterior.mean
            variance = posterior.variance
            var.append(variance.detach().numpy().reshape(1,-1)[0])
            yhat.append(mean.detach().numpy().reshape(1,-1)[0])
            #print(yhat)
                
            #print('New candidate values',candidates,func(candidates))
            ##update the labeled pool
            train_X=torch.cat((train_X,candidates),0)
            train_Y=torch.cat((train_Y,torch.from_numpy(func(candidates.numpy()).reshape(-1,1))),0)
            print(train_X.shape,train_Y.shape)
            #print('Minimum Found So Far',train_Y.max())
            B=B-batch_size
            print('**********')
            
        end=time.time()
        duration=end-start
        print('Timing of run is ',duration,' seconds')
        df = pd.DataFrame(train_X.numpy(), columns=np.arange(1,d+1).tolist())
        df['f(x)'] = train_Y.numpy()
        all_acquisition_values = np.hstack(acq_values).reshape(1, df.shape[0])[0]
        df['acquisition_value'] = all_acquisition_values
        df['mean']= np.hstack(yhat).reshape(1, df.shape[0])[0]
        df['variance']= np.hstack(var).reshape(1, df.shape[0])[0]
        df['BKS']= np.hstack(yvals).reshape(1, df.shape[0])[0]
             
        #Problem name 
        Name=pname
        
        if(acq=='EI'):
            df.to_csv(Name+'_'+str(d)+'d_EI_'+str(i)+'.csv', index=False)
        elif(acq=='UCB'):
            df.to_csv(Name+'_'+str(d)+'d_UCB_'+str(i)+'.csv', index=False)           
        elif(acq=='MES'):
            df.to_csv(Name+'_'+str(d)+'d_MES_'+str(i)+'.csv', index=False)
        elif(acq=='Gibbon'):
            df.to_csv(Name+'_'+str(d)+'d_Gibbon_'+str(i)+'.csv', index=False)

        print('\n ***********************\n ')


#2d:80,256,4,6
#10d:400,1000,8,22
#14d:2000,1400,50,30 
#60d:3000,6000,50,122

# dd=14
# K=50
# BUDGET=2000
# cand_size=100*dd
# init_size=2*(dd+1)
# Name='Robot'

dd=6
K=8
BUDGET=400
cand_size=100*dd
init_size=2*(dd+1)
Name='Levy'

bayopt(acq='EI',budget=BUDGET,pool_size=cand_size,batch_size=K,initial_size=init_size,d=dd,pname=Name)
bayopt(acq='UCB',budget=BUDGET,pool_size=cand_size,batch_size=K,initial_size=init_size,d=dd,pname=Name)
bayopt(acq='Gibbon',budget=BUDGET,pool_size=cand_size,batch_size=K,initial_size=init_size,d=dd,pname=Name)
bayopt(acq='MES',budget=BUDGET,pool_size=cand_size,batch_size=K,initial_size=init_size,d=dd,pname=Name)



# In[ ]:





# In[ ]:





# In[ ]:




