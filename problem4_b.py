import numpy as np
import math
from scipy.constants import alpha
from scipy.stats import norm
from iminuit import Minuit, describe


def mW(mH, mt): #Function to compute W boson mass in function of Higgs mass and top mass

    c = [0, 0.05429, 0.008939, 0.0000890, 0.000161, 1.070, 0.5256, 0.0678, 0.00179, 0.0000659, 0.0737, 114.9]

    dH = np.log(mH/100)
    dh = (mH/100)**2
    d_alpha = (alpha/(1- (314.97e-4 + 276.8e-4 - 0.7e-4)))/0.05907 -1
    dt = (mt/174.3)**2 - 1
    d_alphaS = 0.1176/0.119 - 1
    dZ = 91.1875/91.1875 - 1

    mass_W = 80.3799 - c[1]*dH -c[2]* dH**2 + c[3]* dH**4 + c[4]*(dh-1) - c[5]*d_alpha + c[6]*dt - c[7]*dt**2 + c[8]*dH*dt + c[9]*dh*dt - c[10]*d_alphaS + c[11]*dZ

    return mass_W

def gaussian_W(mH,mt, error): #Gaussian distribution of the W boson mass in function of Higgs boson mass and top mass with error as standard deviation   
    return  np.exp(-np.power(mW(mH, mt) - 81.286, 2.) / (2 * np.power(81.286*error, 2.)))

def gaussian_t(mt, error): #Gaussian distribution of top mass                                                                                             
    return  np.exp(-np.power(mt - 174.3, 2.) / (2 * np.power(mt*error, 2.)))

def nll(mH, mt): #Minimisation of mH and mt
    return -np.log(gaussian_W(mH, mt, error = 0.01)+gaussian_t(mt, error=0.1))

m = Minuit(nll, mH = 125, mt=172.4)
m.limits = [(121, 130), (172, 173)]
m.fixed["mt"] = True
m.errordef = Minuit.LIKELIHOOD
m.migrad()
m.hesse()

mh_min = m.values["mH"] #Stock Higgs boson mass value
mh_err = m.errors[0] #Stock Higgs boson error
print("mH = ", mh_min)
print("error on mH = ", mh_err)

def nll2(mH, mt): #Minimisation of mH and mt
    return -np.log(gaussian_W(mH, mt, error = 0.01)+gaussian_t(mt, error=0.01))

m = Minuit(nll2, mH = 125, mt=172.4)
m.limits = [(121, 130), (172, 173)]
m.fixed["mt"] = True
m.errordef = Minuit.LIKELIHOOD
m.migrad()
m.hesse()

mh_min = m.values["mH"] #Stock Higgs boson mass value
mh_err = m.errors[0] #Stock Higgs boson error  
print("mH = ", mh_min)
print("error on mH = ", mh_err)


def nll3(mH, mt): #Minimisation of mH and mt
    return -np.log(gaussian_W(mH, mt, error = 0.01)+gaussian_t(mt, error=0.001))                                                       

m = Minuit(nll3, mH = 125, mt=172.4)
m.limits = [(121, 130), (172, 173)]
m.fixed["mt"] = True
m.errordef = Minuit.LIKELIHOOD
m.migrad()
m.hesse()

mh_min = m.values["mH"] #Stock Higgs boson mass value
mh_err = m.errors[0] #Stock Higgs boson error  
print("mH = ", mh_min)
print("error on mH = ", mh_err)
