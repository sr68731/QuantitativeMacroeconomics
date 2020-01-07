# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 13:19:46 2019

@author: Sebastian Amit Roy
"""

import numpy as np
import matplotlib as mpl
import quantecon as qe

#Parametrisation and calculation of constants

alpha=0.3
beta=0.99**40
tau=0
lamb=0.5 #since 'lambda' is generic fct in Python
T=50000
Phi=(1+((1-alpha)/(alpha*(1+lamb)))*(lamb+tau))**(-1) #assuming unity shock expectation
s=beta*Phi/(1+(beta*Phi))

#Shock generation

zeta=np.random.lognormal(1, 0.13, (1, 50000))
rho=np.random.lognormal(1, 0.5, (1, 50000))
eta=np.random.lognormal(1, 0.95, (1, 50000))

#Capital path

k=[1]
t=[1]
iter=1

while iter<T:
    k.append(np.exp(np.log((1-alpha)/(1-lamb))+np.log(s)+np.log(1-tau)+np.log(zeta[0, iter])+alpha*np.log(k[iter-1])))
    iter+=1
    t.append(iter)

#Capital path with learning
#Note that now Phi isn't constant anymore

k_learning=[1]
t_learning=[1]
iter=1

while iter<T:
    #Define E_shock as sample mean from iter first periods
    #Using those expectations, compute Phi and savings
    #Generate new capital path
    E_rho=np.mean(rho[0,0:iter])
    E_eta=np.mean(eta[0,0:iter])
    Phi=(1+(1-alpha)/(alpha*(1+lamb)*E_rho)*(lamb*E_eta+tau*(1+lamb*(1-E_eta))))**(-1)
    s=beta*Phi/(1+(beta*Phi))
    k_learning.append(np.exp(np.log((1-alpha)/(1-lamb))+np.log(s)+np.log(1-tau)+np.log(zeta[0, iter])+alpha*np.log(k[iter-1])))
    iter+=1
    t_learning.append(iter)

mean_k=np.mean(k)
mean_k_learning=np.mean(k_learning)

mpl.pyplot.scatter(t, k, s=0.25, label=r'Capital path without learning')
mpl.pyplot.scatter(t, k_learning, s=0.25, label=r'Capital path with learning')
mpl.pyplot.hlines(mean_k, xmin=0, xmax=T, color='red', label=r'Mean without learning')
mpl.pyplot.hlines(mean_k_learning, xmin=0, xmax=T, color='green', label=r'Mean with learning')
mpl.pyplot.xlabel(r'model periods (40 years each)')
mpl.pyplot.ylabel(r'capital per unit of efficient labour')
mpl.pyplot.legend()
mpl.pyplot.title(r'Capital path')
#mpl.pyplot.xlim(left=0, right=100) #uncommenting this line allows for better insight into the simulation structure, as it limits number of points plotted
mpl.pyplot.show()

#Finally, with shock discretisation

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

#Discretised capital path simulation
i=1
ln_kk=0
while i<T:
    ln_kk=np.log(s_disc)+np.log(1-tau)+np.log(1-alpha)+np.log(zeta_draw[i])-np.log(1+lamb)-np.log(1)+alpha*np.log(k)
    kk=np.exp(ln_kk)
    k_prime.append(kk) 
    k=kk
    i=i+1  

t= np.linspace(0,T, T)   

mpl.pyplot.scatter(t,k_prime, s=0.01, label=r'Capital path')
mpl.pyplot.xlabel(r'model periods (40 years each)')
mpl.pyplot.ylabel(r'capital per unit of efficient labour')
mpl.pyplot.hlines(k_prime[0], xmin=0, xmax=T, label=r'Capital steady state', color='red')
mpl.pyplot.ylim(0.016,0.024)
mpl.pyplot.legend(loc='lower right')
mpl.pyplot.title(r'Discretised capital path')
mpl.pyplot.show()











