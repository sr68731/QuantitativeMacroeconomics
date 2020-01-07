# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 23:00:40 2020

@author: Sebastian Amit Roy
"""

# =============================================================================
' Excercise 1.3 - Simple Krussel Smith Method'
# =============================================================================

import numpy as np 
import matplotlib.pyplot as plt
import quantecon as qe
import scipy as sc
from scipy import optimize

T=50000
lamb = 0.5 #lambda
tau= 0.0
alpha=0.3
beta=0.99**40
g=0

#Zeta shock
zeta_disc=[0,0]   
zeta_disc[1]= 1.13
zeta_disc[0]=0.87
probzeta=0.5

#Eta shock
[eta,probeta] = qe.quad.qnwnorm(11, 1, 40*0.15**2)
eta_disc=np.exp(eta)

#Rho shock
rho_disc=[0,0]   
rho_disc[1]= 1.5
rho_disc[0]=0.5
probrho=0.5

Phi_disc=0
state_matrix=np.zeros((11,2))

#For lambda 0.5: 
for i, e in enumerate(eta_disc): 
    for j, r in enumerate(rho_disc):
       state_matrix[i,j]=1/(1+((1-alpha)/alpha*(1+lamb)*r)*(lamb*e+tau*(1+lamb*(1-e))))
       Phi_disc=Phi_disc+probrho*probeta[i]*state_matrix[i,j] #given the probability for occurence of the shock
               
s_disc=beta*Phi_disc/(1+beta*Phi_disc)  

#Getting capital steady state
zeta_draw= np.random.choice(zeta_disc, size=T, p=[.5, .5])
k_prime=[]
ln_k=1/(1-alpha)*(np.log(s_disc)+np.log(1-tau)+np.log(1-alpha)-np.log(1+lamb)-np.log(1))
k=np.exp(ln_k)
k_prime.append(k)

Psi0=np.log(s_disc)+np.log(1-tau)+np.log(1-alpha)-np.log(1+lamb)-np.log(1+g)+np.log(sum(zeta_draw))/len(zeta_draw)
i=1
j=0

Psi1=alpha

#Parameters
state_matrix=np.empty([5,2]) #matrix of states (5 capital states, 2 shock states)
ypsilon=1 #normalisation
sav_rate=np.empty([5,2]) #each state has unique optimal savings
old_lab_inc=0
old_pens_inc=0
pk=0.2 #equally likely capital states

#Function definictions
def R(zeta,rho,k): #gross interest rate
    return alpha*k**(alpha-1)*zeta*rho

def w(k,zeta):  #wage
    w1 = (1-alpha)*ypsilon*k**alpha*zeta
    return w1
def old_lab_inc_func (capital): #old labour income, I
    old_lab_inc=0
    for k,kk  in enumerate(k_grid): #for each today and tomorrow capital
        for e,ee  in enumerate(eta_disc): #for each today and tomorrow eta
            for z, zz in enumerate(zeta_disc): #... shock zeta
                for r, rr in enumerate(rho_disc): #... shock rho
                    b=(ee*(1-alpha)*ypsilon*capital**alpha*zz/(alpha*kk**(alpha-1)*zz*rr))
                    old_lab_inc+=(b*probzeta*probrho*pk*probeta[e])
    return old_lab_inc


def old_pens_inc_func (zeta,capital):    #pension income, II
    old_pens_inc=0
    for k,kk  in enumerate(k_grid):
        b=(tau*(1+lamb/(1-lamb))*w(capital,zeta))/(alpha*kk**(alpha-1)*1*1) 
        old_pens_inc+=(b*pk)
    return old_pens_inc
 
#Grid  
k_nodes=5
k_min=0.5*k_prime[0]
k_max=1.5*k_prime[0]
k_grid= np.linspace(k_min,k_max, k_nodes)   

#Solving the HH problem along the linearised Euler Equation

for k, kk in enumerate(k_grid):
    for j, z in enumerate(zeta_disc):
        def linearised_Euler(a1):
           a=beta*w(k,z)-a1-lamb*(1-tau)*old_lab_inc_func(kk)-(1-lamb)*old_pens_inc_func(z,kk) #Euler after transf., III
           return (a)
        state_matrix[k,j]=optimize.brentq(linearised_Euler,0.0000001,1)
        old_pens_inc=0
        old_lab_inc=0
        sav_rate[k,j]=state_matrix[k,j]/(1-tau)*w(k,z) #Policy function for savings


    
