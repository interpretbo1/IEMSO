#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import csv
import math
import os
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from BBOB_problems import *
from Rover import * 

def Rover60(x):   
    domain = create_large_domain(force_start=False,
                                 force_goal=False,
                                 start_miss_cost=l2cost,
                                 goal_miss_cost=l2cost)
    n_points = domain.traj.npoints
    raw_x_range = np.repeat(domain.s_range, n_points, axis=1)
    f_max = 5.0 # maximum value of f
    f = ConstantOffsetFn(domain, f_max)
    f = NormalizedInputFn(f, raw_x_range)
    x_range = f.get_range()
    
    #n=len(x)
    ans=f(x) #[f(x[i]) for i in range(n)]
    
    return -ans #np.array(ans) 

def rosenbrock(x):
        total = 0
        for i in range(len(x) - 1):
            total += 100 * (x[i] ** 2 - x[i + 1]) ** 2 + (x[i] - 1) ** 2
        return total
    
def rastrigin(x):
    d=len(x)
    total=10 * d + sum(x ** 2 - 10 * np.cos(2 * np.pi * x))
    return total

def levy(x):
    import math
    d=len(x)
    w= 1 + (x - 1)/4
    pi=math.pi
    sin=np.sin
    p2=np.exp2
    term1=(sin(pi*w[0]))**2
    a1=(w[d-1]-1)**2
    a2=1+(sin(2*pi*w[d-1]))**2
    term3=a1*a2
    wi=w[0:(d-1)]
    b1=(wi-1)**2
    b2=1+10*(sin((pi*wi)+1)**2)
    summ=sum(b1*b2)
    y=term1+summ +term3
    
    return y

def Michalewicz(x):
    d=len(x)
    y=-np.sum(np.sin(x) * (np.sin(((1 + np.arange(d)) * x ** 2) / np.pi)) ** 20)
    return y

def Zakharov(x):
    d=len(x)
    y=np.sum(x ** 2) + np.sum(0.5 * (1 + np.arange(d)) * x) ** 2 
    total=y + np.sum(0.5 * (1 + np.arange(d)) * x) ** 4
    return total 

def Perm(x):
    d=len(x)
    beta = 10.0
    outer = 0.0
    for ii in range(d):
        inner = 0.0
        for jj in range(d):
            xj = x[jj]
            inner += ((jj + 1) + beta) * (xj ** (ii + 1) - (1.0 / (jj + 1)) ** (ii + 1))
        outer += inner ** 2
    return outer        

def Ackley(x):
    d=len(x)
    term1=np.exp(-0.2 * np.sqrt(sum(x ** 2) / d))
    term2=np.exp(sum(np.cos(2.0 * np.pi * x)) / d)
    total=-20.0 * term1-term2+ 20+ np.exp(1)
    #print(total)
    return total


def evaluate(x,func):
    xx=x.copy()
    n=x.shape[0]
    m=x.shape[1]
    #print('n is:',n)
    #print(x)
    ys=[]
    if (func==1):
        for i in range(n):
            ys.append(rosenbrock(x.iloc[i,:]))
    elif(func==2):
        for i in range(n):
            ys.append(rastrigin(x.iloc[i,:]))    
    elif(func==3):
        for i in range(n):
            ys.append(levy(x.iloc[i,:]))
    elif(func==4):
        for i in range(n):
            ys.append(Michalewicz(x.iloc[i,:]))
    elif(func==5):
        for i in range(n):
            ys.append(Zakharov(x.iloc[i,:]))
    elif(func==6):
        for i in range(n):
            ys.append(Perm(x.iloc[i,:]))
    elif(func==7 or func==77):
        for i in range(n):
            ys.append(Ackley(x.iloc[i,:]))
    elif(func==10):  
        for i in range(n):
            ys.append(sphere1(x.iloc[i,:]))               
    elif(func==11):  
        for i in range(n):
            ys.append(ellipsoid2(x.iloc[i,:]))               
    elif(func==12):  
        for i in range(n):
            ys.append(rastrigin3(x.iloc[i,:]))               
    elif(func==13):  
        for i in range(n):
            ys.append(buecheRastrigin4(x.iloc[i,:]))               
    elif(func==14):  
        for i in range(n):
            ys.append(linearSlope5(x.iloc[i,:]))               
    elif(func==15):  
        for i in range(n):
            ys.append(attractiveSector6(x.iloc[i,:]))               
    elif(func==16):  
        for i in range(n):
            ys.append(stepEllipsoid7(x.iloc[i,:]))               
    elif(func==17):  
        for i in range(n):
            ys.append(rosenbrock8(x.iloc[i,:]))               
    elif(func==18):  
        for i in range(n):
            ys.append(rosenbrockRotated9(x.iloc[i,:]))               
    elif(func==19):  
        for i in range(n):
            ys.append(ellipsoidRotated10(x.iloc[i,:]))               
    elif(func==20):  
        for i in range(n):
            ys.append(discus11(x.iloc[i,:]))               
    elif(func==21):  
        for i in range(n):
            ys.append(bentCigar12(x.iloc[i,:]))               
    elif(func==22):  
        for i in range(n):
            ys.append(sharpRidge13(x.iloc[i,:]))               
    elif(func==23):  
        for i in range(n):
            ys.append(differentPowers14(x.iloc[i,:]))               
    elif(func==24):  
        for i in range(n):
            ys.append(rastriginRotated15(x.iloc[i,:]))               
    elif(func==25):  
        for i in range(n):
            ys.append(weierstrass16(x.iloc[i,:]))               
    elif(func==26):  
        for i in range(n):
            ys.append(schaffers17(x.iloc[i,:]))               
    elif(func==27):  
        for i in range(n):
            ys.append(schaffers18(x.iloc[i,:]))               
    elif(func==28):  
        for i in range(n):
            ys.append(griewankRosenBrock19(x.iloc[i,:]))               
    elif(func==29):  
        for i in range(n):
            ys.append(schwefel20(x.iloc[i,:]))               
    elif(func==30):  
        for i in range(n):
            ys.append(gallagher21(x.iloc[i,:]))               
    elif(func==31):  
        for i in range(n):
            ys.append(gallagher22(x.iloc[i,:]))               
    elif(func==32):  
        for i in range(n):
            ys.append(katsuura23(x.iloc[i,:]))               
    elif(func==33):  
        for i in range(n):
            ys.append(lunacekBiRastrigin24(x.iloc[i,:]))                            
    elif(func==80):
        for i in range(n):
            ys.append(Rover60(np.array(x.iloc[i,:])))
            
    yy=pd.DataFrame(ys)
    xx[m]=yy.copy()
    xx[m+1]=yy.copy()
    return xx

#Sphere1,Ellipsoid2,Rastrigin3,BuecheRastrigin4, LinearSlope5
#AttractiveSector6, StepEllipsoid7, Rosenbrock8, RosenbrockRotated9
#EllipsoidRotated10, Discus11, BentCigar12, SharpRidge13, DifferentPowers14
#RastriginRotated15, Weierstrass16, Schaffers17, Schaffers18, GriewankRosenBrock19 
#Schwefel20, Gallagher21, Gallagher22, Katsuura23, LunacekBiRastrigin24