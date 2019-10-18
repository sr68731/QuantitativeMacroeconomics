# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 12:41:16 2019

@author: Sebastian Amit Roy
"""

import sympy as sp
import numpy as np
import matplotlib as mpl
import scipy as scp
import time

beta=0.988 #standard US rate of impatience
theta=0.679 #labour share
delta=0.013 #standard US depreciation rate
h=1 #fixed labour supply
kappa=0
#kappa=5.24
nu=2

k_low=5 #set lower bound for the capital grid
k_high=55 #set upper bound for the capital grid
fine=101 #set density of the grid

k_grid=np.linspace(k_low, k_high, fine)

vfpx=np.zeros((1,fine-1)) #initialising x-grid for plotting valfun
for i in range(0, fine-1):
    vfpx[0,i]=k_grid[i]

##############################################################################
#VFI
##############################################################################

tic=time.time()

valfun=np.zeros((1,fine-1)) #create array of zeros to store value function
g=np.zeros((1, fine-1)) #create array to store indices of optimal capital levels (useful for policy function)

#Now we'll build the return matrix:

M=np.zeros((fine-1, fine-1)) #initialize the matrix
for i in range(0, fine-1):
    for j in range(0, fine-1):
        if (k_grid[i]**(1-theta))*(h**theta)-k_grid[j]+(1-delta)*k_grid[i]>=0:
            M[i,j]=np.log((k_grid[i]**(1-theta))*((h)**theta)-k_grid[j]+(1-delta)*k_grid[i])-kappa*(h**(1+(1/nu)))/(1+(1/nu))
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

toc=time.time()

print("Value function iteration required ", iter, " iterations")
print("Value function iterations took ", toc-tic, " seconds")

#Here value function iteration ends

##############################################################################
#Capital path computation
##############################################################################

maxiter=1000
conv=0.001
progress=1
k_init=25 #set the initial value of capital for the convergence path computation
k_path=np.zeros((1, maxiter)) #initialize array that stores capital path
k_path[0, 0]=k_init
iter=1 #number of started iterations
while iter<maxiter and progress>conv:# and progress>conv: #this is to FIND steady state, for simulating capital path delete second condition!
    i=np.where(k_grid==k_init) #find index of k_init in k_grid
    j=int(i[0])
    l=int(g[0,j])
    k_path[0,iter]=k_grid[l] #next element in k_path is element of k_grid whose index is stored in g[j]
    progress=abs(k_path[0,iter]-k_path[0,iter-1])
    k_init=np.copy(k_path[0,iter]) #it's clumsy, but quick to treat k_init as an operative variable
    iter=iter+1
    
print("Capital converged to the steady state after ", iter-1, " iterations")
    
k_ss=k_path[0, iter-1]

##############################################################################
#Valfun plotting
##############################################################################

mpl.pyplot.scatter(vfpx, valfun, s=1, color='blue')
mpl.pyplot.xlabel('capital')
mpl.pyplot.ylabel('value function')
mpl.pyplot.legend()
#mpl.pyplot.xlim([0, 200])
#mpl.pyplot.ylim([0, -500])
mpl.pyplot.title(r"Value function")
mpl.pyplot.show()

##############################################################################
#Decision rule monotonicity
##############################################################################

tic=time.time()

valfun=np.zeros((1,fine-1)) #create array of zeros to store value function
g=np.zeros((1, fine-1)) #create array to store indices of optimal capital levels (useful for policy function)

#Now we'll build the return matrix:

M=np.zeros((fine-1, fine-1)) #initialize the matrix
for i in range(0, fine-1):
    for j in range(0, fine-1):
        if (k_grid[i]**(1-theta))*((h)**theta)-k_grid[j]+(1-delta)*k_grid[i]>=0:
            M[i,j]=np.log((k_grid[i]**(1-theta))*((h)**theta)-k_grid[j]+(1-delta)*k_grid[i])-kappa*(h**(1+1/nu))/(1+1/nu)
        else: M[i,j]=-1000000
        
#Now true iteration starts
        
iter=0 #number of completed iterations
maxiter=10000 #maximum number of iterations
conv=0.001 #convergence level

chi=np.zeros((fine-1, fine-1)) #initialize chi
progress=1

#first iteration goes as it used to go, because we need initial g
valfun_old=np.copy(valfun) #store old valfun
for i in range(0, fine-1):
    chi[i,]=M[i,]+beta*valfun #update chi
for i in range(0, fine-1):
    valfun[0,i]=np.amax(chi[i,]) #update valfun
    g[0,i]=np.argmax(chi[i,])
progress=abs(np.amax(valfun-valfun_old)) #check progress
iter=iter+1

#now we apply monotonicity:
while iter<maxiter and progress>conv:
    valfun_old=np.copy(valfun)
    for i in range (0, fine-1):
        for j in range(int(g[0,i]), fine-1):
            chi[i,j]=M[i,j]+beta*valfun[0,j]
    for i in range(0, fine-1):
        valfun[0,i]=np.amax(chi[i,]) #update valfun
        g[0,i]=np.argmax(chi[i,])
    progress=abs(np.amax(valfun-valfun_old)) #check progress
    iter=iter+1
            

toc=time.time()

mpl.pyplot.scatter(vfpx, valfun, s=1, color='blue')
mpl.pyplot.xlabel('capital')
mpl.pyplot.ylabel('value function')
mpl.pyplot.legend()
#mpl.pyplot.xlim([0, 200])
#mpl.pyplot.ylim([0, -500])
mpl.pyplot.title(r"Value function under g-monotonicity")
mpl.pyplot.show()

print("Value function iteration required ", iter-1, " iterations")
print("Value function iterations took ", toc-tic, " seconds")

##############################################################################
#Value function concavity
##############################################################################

tic=time.time()

valfun=np.zeros((1,fine-1)) #create array of zeros to store value function
g=np.zeros((1, fine-1)) #create array to store indices of optimal capital levels (useful for policy function)

#Now we'll build the return matrix:

M=np.zeros((fine-1, fine-1)) #initialize the matrix
for i in range(0, fine-1):
    for j in range(0, fine-1):
        if (k_grid[i]**(1-theta))*((h)**theta)-k_grid[j]+(1-delta)*k_grid[i]>=0:
            M[i,j]=np.log((k_grid[i]**(1-theta))*((h)**theta)-k_grid[j]+(1-delta)*k_grid[i])-kappa*(h**(1+1/nu))/(1+1/nu)
        else: M[i,j]=-1000000
        
#Now true iteration starts
        
iter=0 #number of completed iterations
maxiter=1000 #maximum number of iterations
conv=0.001 #convergence level

chi=np.zeros((fine-1, fine-1)) #initialize chi
progress=1

#first iteration goes as it used to go, because we need initial g
valfun_old=np.copy(valfun) #store old valfun
for i in range(0, fine-1):
    chi[i,]=M[i,]+beta*valfun #update chi
for i in range(0, fine-1):
    valfun[0,i]=np.amax(chi[i,]) #update valfun
    g[0,i]=np.argmax(chi[i,])
progress=abs(np.amax(valfun-valfun_old)) #check progress
iter=iter+1

while iter<maxiter and progress>conv:
#for j in range(0, maxiter):
    valfun_old=np.copy(valfun)
    for i in range(0, fine-1):
        for j in range(0, fine-1):
            chi[i,j]=M[i,j]+beta*valfun[0,j]
            if j>0 and chi[i,j]<chi[i,j-1]:
                #g[0,i]=j
                break
            valfun[0,i]=np.amax(chi[i,]) #update valfun
            g[0,i]=np.argmax(chi[i,])
    progress=abs(np.amax(valfun-valfun_old))
    iter=iter+1

toc=time.time()

mpl.pyplot.scatter(vfpx, valfun, s=1, color='blue')
mpl.pyplot.xlabel('capital')
mpl.pyplot.ylabel('value function')
mpl.pyplot.legend()
#mpl.pyplot.xlim([0, 200])
#mpl.pyplot.ylim([0, -500])
mpl.pyplot.title(r"Value function under valfun-concavity")
mpl.pyplot.show()

print("Value function iteration required ", iter, " iterations")
print("Value function iterations took ", toc-tic, " seconds")

##############################################################################
#Decision rule monotonicity and value function concavity
##############################################################################

tic=time.time()

valfun=np.zeros((1,fine-1)) #create array of zeros to store value function
g=np.zeros((1, fine-1)) #create array to store indices of optimal capital levels (useful for policy function)

#Now we'll build the return matrix:

M=np.zeros((fine-1, fine-1)) #initialize the matrix
for i in range(0, fine-1):
    for j in range(0, fine-1):
        if (k_grid[i]**(1-theta))*((h)**theta)-k_grid[j]+(1-delta)*k_grid[i]>=0:
            M[i,j]=np.log((k_grid[i]**(1-theta))*((h)**theta)-k_grid[j]+(1-delta)*k_grid[i])-kappa*(h**(1+1/nu))/(1+1/nu)
        else: M[i,j]=-1000000
        
#Now true iteration starts
        
iter=0 #number of completed iterations
maxiter=1000 #maximum number of iterations
conv=0.001 #convergence level

chi=np.zeros((fine-1, fine-1)) #initialize chi
progress=1

#first iteration goes as it used to go, because we need initial g
valfun_old=np.copy(valfun) #store old valfun
for i in range(0, fine-1):
    chi[i,]=M[i,]+beta*valfun #update chi
for i in range(0, fine-1):
    valfun[0,i]=np.amax(chi[i,]) #update valfun
    g[0,i]=np.argmax(chi[i,])
progress=abs(np.amax(valfun-valfun_old)) #check progress
iter=iter+1

while iter<maxiter and progress>conv:
#for j in range(0, maxiter):
    valfun_old=np.copy(valfun)
    for i in range(0, fine-1):
        for j in range(int(g[0,i]), fine-1):
            chi[i,j]=M[i,j]+beta*valfun[0,j]
            if j>0 and chi[i,j]<chi[i,j-1]:
                #g[0,i]=j
                break
            valfun[0,i]=np.amax(chi[i,]) #update valfun
            g[0,i]=np.argmax(chi[i,])
    progress=abs(np.amax(valfun-valfun_old))
    iter=iter+1

toc=time.time()

mpl.pyplot.scatter(vfpx, valfun, s=1, color='blue')
mpl.pyplot.xlabel('capital')
mpl.pyplot.ylabel('value function')
mpl.pyplot.legend()
#mpl.pyplot.xlim([0, 200])
#mpl.pyplot.ylim([0, -500])
mpl.pyplot.title(r"Value function under g-monotonicity and valfun-concavity")
mpl.pyplot.show()

print("Value function iteration required ", iter, " iterations")
print("Value function iterations took ", toc-tic, " seconds")

##############################################################################
#Local search
##############################################################################

tic=time.time()

local=5 #define bounds for the local search

valfun=np.zeros((1,fine-1)) #create array of zeros to store value function
g=np.zeros((1, fine-1)) #create array to store indices of optimal capital levels (useful for policy function)

#Now we'll build the return matrix:

M=np.zeros((fine-1, fine-1)) #initialize the matrix
for i in range(0, fine-1):
    for j in range(0, fine-1):
        if (k_grid[i]**(1-theta))*((h)**theta)-k_grid[j]+(1-delta)*k_grid[i]>=0:
            M[i,j]=np.log((k_grid[i]**(1-theta))*((h)**theta)-k_grid[j]+(1-delta)*k_grid[i])-kappa*(h**(1+1/nu))/(1+1/nu)
        else: M[i,j]=-1000000
        
#Now true iteration starts
        
iter=0 #number of completed iterations
maxiter=1000 #maximum number of iterations
conv=0.001 #convergence level

chi=np.zeros((fine-1, fine-1)) #initialize chi
progress=1

while iter<maxiter and progress>conv:
    valfun_old=np.copy(valfun) #store old valfun
    for i in range(0, local-1):
        chi[i,]=M[i,]+beta*valfun #update chi
    for i in range(local, fine-1-local):
        for j in range(i-local, i+local):
            chi[i,j]=M[i,j]+beta*valfun[0,j]
    for i in range(fine-local, fine-1):
        chi[i,]=M[i,]+beta*valfun #update chi
    for i in range(0, fine-1):
        valfun[0,i]=np.amax(chi[i,]) #update valfun
        g[0,i]=np.argmax(chi[i,])
    progress=abs(np.amax(valfun-valfun_old)) #check progress
    iter=iter+1

toc=time.time()

mpl.pyplot.scatter(vfpx, valfun, s=1, color='blue')
mpl.pyplot.xlabel('capital')
mpl.pyplot.ylabel('value function')
mpl.pyplot.legend()
#mpl.pyplot.xlim([0, 200])
#mpl.pyplot.ylim([0, -500])
mpl.pyplot.title(r"Value function under local search")
mpl.pyplot.show()

print("Value function iteration required ", iter, " iterations")
print("Value function iterations took ", toc-tic, " seconds")

##############################################################################
#Howard method
##############################################################################

tic=time.time()

valfun=np.zeros((1,fine-1)) #create array of zeros to store value function
g=np.zeros((1, fine-1)) #create array to store indices of optimal capital levels (useful for policy function)

#Now we'll build the return matrix:

M=np.zeros((fine-1, fine-1)) #initialize the matrix
for i in range(0, fine-1):
    for j in range(0, fine-1):
        if (k_grid[i]**(1-theta))*(h**theta)-k_grid[j]+(1-delta)*k_grid[i]>=0:
            M[i,j]=np.log((k_grid[i]**(1-theta))*((h)**theta)-k_grid[j]+(1-delta)*k_grid[i])-kappa*(h**(1+(1/nu)))/(1+(1/nu))
        else: M[i,j]=-1000000
        
#Now true iteration starts
        
iter=0 #number of completed iterations
maxiter=1000 #maximum number of iterations
conv=0.001 #convergence level

chi=np.zeros((fine-1, fine-1)) #initialize chi
progress=1

while iter<maxiter and progress>conv:
    valfun_old=np.copy(valfun) #store old valfun
    if iter<200:
        for i in range(0, fine-1):
            chi[i,]=M[i,]+beta*valfun #update chi
        for i in range(0, fine-1):
            valfun[0,i]=np.amax(chi[i,]) #update valfun
        progress=abs(np.amax(valfun-valfun_old))
        iter=iter+1
    else:
        for i in range(0, fine-1):
            chi[i,]=M[i,]+beta*valfun #update chi
        for i in range(0, fine-1):
            valfun[0,i]=np.amax(chi[i,]) #update valfun
            g[0,i]=np.argmax(chi[i,])
        progress=abs(np.amax(valfun-valfun_old))
        iter=iter+1

toc=time.time()

print("Value function iteration required ", iter, " iterations")
print("Value function iterations took ", toc-tic, " seconds")

mpl.pyplot.scatter(vfpx, valfun, s=1, color='blue')
mpl.pyplot.xlabel('capital')
mpl.pyplot.ylabel('value function')
mpl.pyplot.legend()
mpl.pyplot.title(r"Value function with Howard method, policy iterations from iter=200")
mpl.pyplot.show()

##############################################################################
#Howard method with regular policy improvements
##############################################################################

tic=time.time()

valfun=np.zeros((1,fine-1)) #create array of zeros to store value function
g=np.zeros((1, fine-1)) #create array to store indices of optimal capital levels (useful for policy function)

#Now we'll build the return matrix:

M=np.zeros((fine-1, fine-1)) #initialize the matrix
for i in range(0, fine-1):
    for j in range(0, fine-1):
        if (k_grid[i]**(1-theta))*(h**theta)-k_grid[j]+(1-delta)*k_grid[i]>=0:
            M[i,j]=np.log((k_grid[i]**(1-theta))*((h)**theta)-k_grid[j]+(1-delta)*k_grid[i])-kappa*(h**(1+(1/nu)))/(1+(1/nu))
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
        if iter%50==0:
            g[0,i]=np.argmax(chi[i,])
    progress=abs(np.amax(valfun-valfun_old)) #check progress
    iter=iter+1

toc=time.time()

print("Value function iteration required ", iter, " iterations")
print("Value function iterations took ", toc-tic, " seconds")

mpl.pyplot.scatter(vfpx, valfun, s=1, color='blue')
mpl.pyplot.xlabel('capital')
mpl.pyplot.ylabel('value function')
mpl.pyplot.legend()
mpl.pyplot.title(r"Value function with Howard method, one policy iteration per 50 iterations")
mpl.pyplot.show()











