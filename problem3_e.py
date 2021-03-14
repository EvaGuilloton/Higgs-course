import numpy as np
import matplotlib.pyplot as plt
import random


# Constants #

m_e = 0.511 # MeV
F = 0. 

# Lists #

mass_h = np.linspace(2, 50, 49) #Higgs mass (in MeV)
nEvents = [] #Number of events detected by the calorimeter

def accept_higgs(h_mass): #Acceptance

    lifetime = 4e-9/h_mass
    #    lifetime = []                                                                                 
    #    for i in range(len(h_mass)):                                                                                                     
    #        x  = (alpha_h/2*h_mass*(1-(4*m_e**2/h_mass**2))**(3/2))**(-1) #Lifetime in function of Higgs mass
    #        lifetime.append(x)
    beta_gamma = np.sqrt(1600**2 - h_mass**2)/h_mass #p = beta*gamma*m = sqrt(E^2 + m^2)                                             
    x_moy = beta_gamma*3e8*lifetime
    acceptance_h = np.exp(-2/x_moy)
    return acceptance_h

def dsigma_dz(screening = False) : #Derivation of the cross section in function of the acceptance

    Z = 82 # Pb atomic number                                                                                                                
    if screening == True:
        F = np.log(184*Z**(-1/3))
    if screening == False:
        F = 1/2
    alphaH = m_e**2 * 1.1663787e-5 * 1e-6 * np.sqrt(2)/(4*np.pi)
    sigma = 2*(1/137)**2 * alphaH*Z**2/m_e**2 * F
    return sigma

def e_energy(mass_H): #Energy of electron in the calorimeter

    gamma = 1600/mass_H # E = gamma*m                                                                                                   
    beta = np.sqrt(1600**2 - mass_H**2)/1600 # beta = p/E                                                                                 
    E_rest = mass_H/2 #Energy at rest
    theta = random.uniform(0, np.pi/10) #Random theta coordinates
    phi = random.uniform(0, np.pi*2) #Random phi coordinates
    p_total = np.sqrt(E_rest**2 - m_e**2) #Value of the momentum
    p_x = p_total*np.sin(theta)*np.cos(phi) #Value of the momentum in x coordinate
    p_z = p_total*np.cos(theta) #Value of the momentum in z coordinate  
    p_z_lab = gamma*p_z + E_rest*beta*gamma #Value of the momentum in z coordinate in the referential of the laboratory 
    E_lab = gamma*E_rest + p_x*beta*gamma #Energy of the electron in the referential of the laboratory
    theta_lab = np.arctan(p_x/p_z_lab) #Theta coordinates in the referential of the laboratory
    return E_lab


diff_sigma = dsigma_dz(False)
N = diff_sigma*2e16 #Number of Higgs created

for m in range(1, 50):
    energy = [] #List to stock the energy of the electrons detected by the calorimeter 
    n = accept_higgs(m+1)*N #Number of Higgs bosons decaying after 2m
    for i in range(int(n)): #For each Higgs bosons
        e = e_energy(m+1)/1e3 #Energy of the electrons (in GeV)
        e2 = np.random.normal(e, 0.107*np.sqrt(e)) #Energy of electrons with a random deviation within 0.107*np.sqrt(e) range
        if e2 > 0.75: energy.append(e2) #If the electrons has enough energy to be detected by the calorimeter
    nEvents.append(len(energy)) #Add the number of events detected by the calorimeter


plt.clf() #Close previous plot canvas
plt.figure() #Open a new canvas                                                                                      
plt.plot(mass_h,nEvents,color='red') #Plot the number of events detected by the calorimeter in function of the Higgs mass                 
plt.xlabel("Higgs mass (MeV)")
plt.ylabel("Number of events detected by the calorimeter")
plt.savefig("nb_events_passing_pb3.pdf")
plt.show()
