# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 16:32:17 2019

@author: Sebastian Amit Roy
"""

import sympy as sp
import numpy as np
import matplotlib as mpl

beta=0.99 #standard US rate of impatience
theta=0.67 #labour share
delta=0.025 #standard US depreciation rate
h=0.31 #fixed labour supply
z=1 #naive guess for technology

k_low=0 #set lower bound for the capital grid
k_high=50 #set upper bound for the capital grid
fine=1001 #set density of the grid

k_grid=np.linspace(k_low, k_high, fine)

valfun=np.zeros((1,fine-1)) #create array of zeros to store value function
g=np.zeros((1, fine-1)) #create array to store indices of optimal capital levels (useful for policy function)

#Now we'll build the return matrix:

M=np.zeros((fine-1, fine-1)) #initialize the matrix
for i in range(0, fine-1):
    for j in range(0, fine-1):
        if (k_grid[i]**(1-theta))*((z*h)**theta)-k_grid[j]+(1-delta)*k_grid[i]>=0:
            M[i,j]=np.log(k_grid[i]**(1-theta)*(z*h)**theta-k_grid[j]+(1-delta)*k_grid[i])
        else: M[i,j]=-1000000
        
#Now true iteration starts
        
iter=0 #number of completed iterations
maxiter=1000 #maximum number of iterations
conv=0.001 #convergence level

chi=np.zeros((fine-1, fine-1)) #initialize chi
progress=1

while iter<maxiter and progress>conv:
#for j in range(0, maxiter):
    valfun_old=np.copy(valfun) #store old valfun
    for i in range(0, fine-1):
        chi[i,]=M[i,]+beta*valfun #update chi
    for i in range(0, fine-1):
        valfun[0,i]=np.amax(chi[i,]) #update valfun
        g[0,i]=np.argmax(chi[i,])
    progress=abs(np.amax(valfun-valfun_old)) #check progress
    iter=iter+1

##############################################################################
#Here value function iteration ends
##############################################################################

#compute steady state level of capital

maxiter=1000
conv=0.001
progress=1
k_init=25 #set the initial value of capital for the convergence path computation
k_path=np.zeros((1, maxiter)) #initialize array that stores capital path
k_path[0, 0]=k_init
iter=1 #number of started iterations
while iter<maxiter and progress>conv: #this is to FIND steady state, for simulating capital path delete second condition!
    i=np.where(k_grid==k_init) #find index of k_init in k_grid
    j=int(i[0])
    l=int(g[0,j])
    k_path[0,iter]=k_grid[l] #next element in k_path is element of k_grid whose index is stored in g[j]
    progress=abs(k_path[0,iter]-k_path[0,iter-1])
    k_init=np.copy(k_path[0,iter]) #it's clumsy, but quick to treat k_init as an operative variable
    iter=iter+1

    
