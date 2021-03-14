import numpy as np
import math
from scipy.constants import alpha
from scipy.stats import norm
from iminuit import Minuit, describe

def mW(mH): #Compute W boson mass

    c = [0, 0.05429, 0.008939, 0.0000890, 0.000161, 1.070, 0.5256, 0.0678, 0.00179, 0.0000659, 0.0737, 114.9]

    dH = np.log(mH/100)
    dh = (mH/100)**2
    d_alpha = (alpha/(1- (314.97e-4 + 276.8e-4 - 0.7e-4)))/0.05907 -1
    dt = (172.4/174.3)**2 - 1
    d_alphaS = 0.1176/0.119 - 1
    dZ = 91.1875/91.1875 - 1

    mass_W = 80.3799 - c[1]*dH -c[2]* dH**2 + c[3]* dH**4 + c[4]*(dh-1) - c[5]*d_alpha + c[6]*dt - c[7]*dt**2 + c[8]*dH*dt + c[9]*dh*dt - c[10]*d_alphaS + c[11]*dZ

    return mass_W

def gaussian_W(mH, error): #Gaussian distribution of W boson mass
    return  np.exp(-np.power(mW(mH) - 81.286, 2.) / (2 * np.power(81.286*error, 2.)))

def mixing_angle(mH): #Compute mixing angle
    
    s0 = 0.2312527
    d = [0, 4.729e-4, 2.07e-5, 3.85e-6, -1.85e-6, 0.0207, -0.002851, 1.82e-4, -9.74e-6, 3.98e-4, -0.655]
    lH = np.log(mH/100)
    delta_H = mH/100
    delta_alpha = (alpha/(1- (314.97e-4 + 276.8e-4 - 0.7e-4)))/0.05907 - 1
    delta_alphaS = 0.1176/0.117 - 1
    delta_t = (172.4/178)**2 - 1
    delta_Z = 91.1875/91.1876 - 1

    angle = s0 + d[1]*lH + d[2]*lH**2 + d[3]*lH**4 + d[4]*(delta_H**2 - 1) + d[5]*delta_alpha + d[6]*delta_t + d[7]*delta_t**2 + d[8]*delta_t*(delta_H - 1) + d[9]*delta_alphaS + d[10]*delta_Z

    return angle

def mW_doublepar(mH, mt): #Function to compute W boson mass in function of Higgs mass and top mass                                         

    c = [0, 0.05429, 0.008939, 0.0000890, 0.000161, 1.070, 0.5256, 0.0678, 0.00179, 0.0000659, 0.0737, 114.9]

    dH = np.log(mH/100)
    dh = (mH/100)**2
    d_alpha = (alpha/(1- (314.97e-4 + 276.8e-4 - 0.7e-4)))/0.05907 -1
    dt = (mt/174.3)**2 - 1
    d_alphaS = 0.1176/0.119 - 1
    dZ = 91.1875/91.1875 - 1

    mass_W = 80.3799 - c[1]*dH -c[2]* dH**2 + c[3]* dH**4 + c[4]*(dh-1) - c[5]*d_alpha + c[6]*dt - c[7]*dt**2 + c[8]*dH*dt + c[9]*dh*dt - c[10]*d_alphaS + c[11]*dZ

    return mass_W


def gaussian_W_doublepar(mH,mt, error): #Gaussian distribution of the W boson mass in function of Higgs boson mass and top mass with error as standard deviation                                                                                                    
    return  np.exp(-np.power(mW_doublepar(mH, mt) - 81.286, 2.) / (2 * np.power(81.286*error, 2.)))

def gaussian_angle(mH, error): #Gaussian distribution of mixing angle                                 
    return  np.exp(-np.power(mixing_angle(mH) - 0.21356, 2.) / (2 * np.power(error, 2.)))

def gaussian_t(mt, error): #Gaussian distribution of top mass                        
    return  np.exp(-np.power(mt - 174.3, 2.) / (2 * np.power(mt*error, 2.)))

def nll(mH): #Minimazing Higgs boson mass
    return -np.log(gaussian_W(mH, error = 0.01)+gaussian_angle(mH, error=1e-3))

m = Minuit(nll, mH = 125)
m.limits = [(121, 130)]
m.errordef = Minuit.LIKELIHOOD
m.migrad()
m.hesse()

mh_min = m.values["mH"] #Stocking Higgs boson mass value
mh_err = m.errors[0] #Stocking Higgs boson mass error
print("mH = ", mh_min)
print("error on mH = ", mh_err)

def nll2(mH, mt): #Minimazing Higgs boson mass
    return -np.log(gaussian_W_doublepar(mH, mt, error = 0.01)+gaussian_t(mt, error=0.01)+gaussian_angle(mH, error=1e-3))

m = Minuit(nll2, mH = 125, mt=172.4)
m.limits = [(121, 130), (172, 173)]
m.fixed["mt"] = True
m.errordef = Minuit.LIKELIHOOD
m.migrad()
m.hesse()

mh_min = m.values["mH"] #Stocking Higgs boson mass value
mh_err = m.errors[0] #Stocking Higgs boson mass error
print("mH = ", mh_min)
print("error on mH = ", mh_err)
