import numpy as np
import math
from scipy.constants import alpha
from scipy.stats import norm
from iminuit import Minuit, describe

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

def gaussian_angle(mH, error): #Gaussian distribution of the mixing angle with error as standard deviation
    return  np.exp(-np.power(mixing_angle(mH) - 0.21356, 2.) / (2 * np.power(error, 2.)))

def nll(mH): #Minimazing mixing angle
    return -np.log(gaussian_angle(mH, error = 1e-3))

m = Minuit(nll, mH = 125)
m.limits = [(121, 130)]
m.errordef = Minuit.LIKELIHOOD
m.migrad()
m.hesse()

mh_min = m.values["mH"] #Stocking Higgs boson mass value 
mh_err = m.errors[0] #Stocking Higgs boson mass error
print("mH = ", mh_min)
print("error on mH = ", mh_err)

def nll2(mH): #Minimazing mixing angle
    return -np.log(gaussian_angle(mH, error = 1e-4))

m = Minuit(nll2, mH = 125)
m.limits = [(121, 130)]
m.errordef = Minuit.LIKELIHOOD
m.migrad()
m.hesse()

mh_min = m.values["mH"] #Stocking Higgs boson mass value
mh_err = m.errors[0] #Stocking Higgs boson mass error 
print("mH = ", mh_min)
print("error on mH = ", mh_err)

def nll3(mH): #Minimazing mixing angle
    return -np.log(gaussian_angle(mH, error = 1e-5))
    

m = Minuit(nll3, mH = 125)
m.limits = [(121, 130)]
m.errordef = Minuit.LIKELIHOOD
m.migrad()
m.hesse()

val_mh = m.values["mH"] #Stocking Higgs boson mass value
err_mh = m.errors[0] #Stocking Higgs boson mass error 
print("mH = ", val_mh)
print("error on mH = ", err_mh)
