import numpy as np
import matplotlib.pyplot as plt
import random

# Constant #
m_e = 0.511 #MeV                                                       
G_f = 1.166e-5 * 1e-6 #MeV^-2 Fermi's constant                   
#alpha_h = m_e**2*G_f*np.sqrt(2)/4/np.pi
F = 0.

# Lists #
mass_h = np.linspace(2, 50, 49) #Higgs mass
lifetime = 4e-9/mass_h #Approximation of lifetime
#lifetime = []
#for i in range(len(mass_h)):
#    x = (alpha_h/2*mass_h*(1-(4*m_e**2/mass_h**2))**(3/2))**(-1) #Lifetime in function of Higgs mass
#    lifetime.append(x)   

def accept_higgs(h_mass): #Acceptance

#    alpha_h = m_e**2*G_f*np.sqrt(2)/4/np.pi
    lifetime_tau = 4e-9/h_mass
#    lifetime_tau = []
#    for i in range(len(h_mass)):
#        x  = (alpha_h/2*h_mass*(1-(4*m_e**2/h_mass**2))**(3/2))**(-1) #Lifetime in function of Higgs mass
#        lifetime_tau.append(x)
    beta_gamma = np.sqrt(1600**2 - h_mass**2)/h_mass #p = beta*gamma*m = sqrt(E^2 + m^2)         
    x_moy = beta_gamma*3e8*lifetime_tau
    return np.exp(-2/x_moy)

def dsigma_dz(screening = False): #Derivation of the cross section by the acceptance

    Z = 82 # Pb atomic number                                                                                  
    if screening == True:
        F = np.log(184*Z**(-1/3))
    if screening == False:
        F = 1/2
    alpha_H = m_e**2 * G_f * np.sqrt(2)/(4*np.pi)
    sigma = 2*(1/137)**2 * alpha_H*Z**2/m_e**2 * F #Derivation of the cross section by the acceptance
    return sigma

diff_sigma = dsigma_dz(False)

N = diff_sigma*2e16 #Number of Higgs created
print(N)
nEvents = accept_higgs(mass_h)*N #Number of Higgs not decayed within 2m 

plt.clf() #Close previous plot canvas                                                                      
plt.figure() #Open a new canvas                                                                                       
plt.plot(mass_h,nEvents,color='red') #Plot the number of events in function of Higgs mass                                                 
plt.xlabel("Higgs mass (MeV)")
plt.ylabel(r'$N(x=2)$')
plt.savefig("events_expected_pb3.pdf")
plt.show()
