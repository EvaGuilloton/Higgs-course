import numpy as np
import matplotlib.pyplot as plt
import random

# Constants #
m_e = 0.511 #MeV
G_f = 1.166e-5 * 1e-6 #MeV^-2 Fermi's constant
alpha_h = m_e**2*G_f*np.sqrt(2)/4/np.pi

# Lists #
mass_h = np.linspace(1, 50, 50) #Higgs mass
#lifetime = []                                                                                                 
#for i in range(len(mass_h)):                                                                                             
#    x = (alpha_h/2*mass_h*(1-(4*m_e**2/mass_h**2))**(3/2))**(-1) #Lifetime in function of Higgs mass                          
#    lifetime.append(x)
lifetime = 4e-9/mass_h #Approximation of lifetime in function of Higgs mass

def accept_higgs(h_mass): #Acceptance

    alpha_H = m_e**2*G_f*np.sqrt(2)/4/np.pi 
    lifetime = 4e-9/h_mass
    #lifetime = []                                                                                    
    #for i in range(len(masses)):                                                                                                       
    #    x = (alpha_h/2*masses*(1-(4*m_e**2/masses**2))**(3/2))**(-1) #Lifetime in function of Higgs mass                               
    #    lifetime.append(x)
    beta_gamma = np.sqrt(1600**2 - h_mass**2)/h_mass #p = beta*gamma*m            
    x_moy = beta_gamma*3e8*lifetime #Average distance where Higgs will decays
    acceptance_h = np.exp(-2/x_moy) #Acceptance in 2 meters
    return acceptance_h

acceptance = accept_higgs(mass_h)


plt.clf() #Close previous plot canvas                                                                                                   
plt.figure() #Open a new canvas                                                                                                           
plt.plot(lifetime,acceptance,color='red') #Plot the acceptance in function of the lifetime                           
plt.xlabel("Lifetime (s)")
plt.ylabel(r'$N(x=2)/N_{0}$')
plt.savefig("lifetime_acceptance_pb3.pdf")
plt.show()

plt.clf() #Close previous plot canvas                 
plt.figure() #Open a new canvas                                                        
plt.plot(mass_h,acceptance,color='red') #Plot the acceptance in function of the mass                   
plt.xlabel("Higgs mass (MeV)")
plt.ylabel(r'$N(x=2)/N_{0}$')
plt.savefig("mass_acceptance_pb3.pdf")
plt.show()
