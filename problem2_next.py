from iminuit import Minuit
import numpy as np
import math
from matplotlib import pyplot as plt

def poisson(mu, k): #Poisson's distribution
    return (mu**(k))*np.exp(-1*mu)/(math.factorial(k))

def likehood(mu_vbf, mu_ggf): #Likehood function
    p1 = poisson(par[0]*mu_vbf + par[1]*mu_ggf + par[2], par[3])
    p2 = poisson(par[4]*mu_vbf + par[5]*mu_ggf + par[6], par[7])
    return -1*np.log(p1 +p2)

par = [0.9, 16.2, 5.2, 24, 4.2, 2.1, 0.9, 8] #par = [n_vbf_SR1, n_ggf_SR1, n_b_SR1, N_SR1, n_vbf_SR2, n_ggfSR2, n_b_SR2, N_SR2]

m = Minuit(likehood, mu_vbf = 1.0, mu_ggf = 1.0) 
m.limits = [(-10.0, 10.0), (-10.0, 10.0)]
m.fixed["mu_ggf"] = False #To fix mu_ggf value
m.fixed["mu_vbf"] = True #To fix mu_vbf value
m.errordef = Minuit.LIKELIHOOD

m.migrad()
value_vbf,value_ggf = m.values #Stock fitted mu_vbf and mu_ggf values
print(m.values)

m.hesse()
error_vbf,error_ggf = m.errors #Stock fitted mu_vbf and mu_ggf errors
print(m.errors)

if m.fixed["mu_ggf"] == True :
    mu_vbf_distr,entries_vbf = m.draw_profile("mu_vbf")

    plt.clf() #Close previous plot canvas                             
    plt.figure() #Open a new canvas                                  
    plt.plot(mu_vbf_distr,entries_vbf,color='red') #Plot minimisation of mu_vbf  
    plt.xlabel("mu_vbf")
    plt.ylabel("")
    plt.vlines(value_vbf,1.25,2.75, colors="black",linestyles='solid')
    plt.vlines(value_vbf-error_vbf,1.25,2.75, colors="black",linestyles='dashed')
    plt.vlines(value_vbf+error_vbf,1.25,2.75, colors="black",linestyles='dashed')
    plt.savefig("mu_vbf_separate.pdf")
    plt.show()
    
elif m.fixed["mu_vbf"] == True :
    mu_ggf_distr,entries_ggf = m.draw_profile("mu_ggf")

    plt.clf() #Close previous plot canvas                               
    plt.figure() #Open a new canvas                                     
    plt.plot(mu_ggf_distr,entries_ggf,color='red') #Plot minimisation of ggf      
    plt.xlabel("mu_ggf")
    plt.ylabel("")
    plt.vlines(value_ggf,1.25,2.75, colors="black",linestyles='solid')
    plt.vlines(value_ggf-error_ggf,1.25,2.75, colors="black",linestyles='dashed')
    plt.vlines(value_ggf+error_ggf,1.25,2.75, colors="black",linestyles='dashed')
    plt.savefig("mu_ggf_separate.pdf")
    plt.show()
