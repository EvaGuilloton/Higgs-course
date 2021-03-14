import numpy as np
import matplotlib.pyplot as plt
import random

# Constants #
m_e = 0.511 # MeV                                                                                                                            

# Lists #
energy = []

def e_energy(mass_H): 

    gamma = 1600/mass_H # E = gamma*m
    beta = np.sqrt(1600**2 - mass_H**2)/1600 # beta = p/E                                                                             
    E_rest = mass_H/2 #Energy of the electrons at rest
    theta = random.uniform(0, np.pi/10) #Random theta coordinates
    phi = random.uniform(0, np.pi*2) #Random phi coordinates      
    p_total = np.sqrt(E_rest**2 - m_e**2) #Value of the momentum
    p_x = p_total*np.sin(theta)*np.cos(phi) #value of the momentum in the x coordinates                                                      
    p_z = p_total*np.cos(theta) #value of the momentum in the z coordinates                                                 
    p_z_lab = gamma*p_z + E_rest*beta*gamma #value of the momentum in the z coordinates in the referential of the laboratory 
    E_lab = gamma*E_rest + p_x*beta*gamma #energy of the electron in the referential of the laboratory
    return E_lab


for i in range(10000): #For 10000 initial events
    energy.append(e_energy(50)/1e3) #List of electron energy (in GeV) for Higgs mass of 50MeV                                 

plt.clf() #Close previous plot canvas                                                                                          
plt.figure() #Open a new canvas                                                                                                         
plt.hist(energy,bins = 100,color="blue", edgecolor="black") #Distribution of electron energy                                   
plt.xlabel("Energy [Gev]")
plt.ylabel("Entries")
plt.annotate("mH = 50MeV",xy=(0.025, 0.8), xycoords='axes fraction')
plt.savefig("energy_pb3_50MeV_without.pdf")
plt.show()



energy = [] #Reinitialisation of the list
for i in range(10000): #For 10000 initial events
    energy.append(e_energy(10)/1e3) #List of electron energy (in GeV) for Higgs mass of 10 MeV


plt.clf() #Close previous plot canvas  
plt.figure() #Open a new canvas 
plt.hist(energy,bins = 100,color="blue", edgecolor="black") #Distribution of electron energy
plt.xlabel("Energy [Gev]")
plt.ylabel("Entries")
plt.annotate("mH = 10 MeV",xy=(0.025, 0.8), xycoords='axes fraction')
plt.savefig("energy_pb3_10MeV_without.pdf")
plt.show()


energy = [] #Reinitialisation of the list
for i in range(10000): #For 10000 initial events
    energy.append(e_energy(2)/1e3) #List of electron energy (in GeV) for Higgs mass of 2MeV

plt.clf() #Close previous plot canvas   
plt.figure() #Open a new canvas                     
plt.hist(energy,bins = 100,color="blue", edgecolor="black") #Distribution of electron energy   
plt.xlabel("Energy [Gev]")
plt.ylabel("Entries")
plt.annotate("mH = 2 MeV",xy=(0.025, 0.8), xycoords='axes fraction')
plt.savefig("energy_pb3_2MeV_without.pdf")
plt.show()

energy = [] #Reinitialisation of the list
for i in range(10000): #For 10000 initial events
    e = e_energy(50)/1e3 #List of electron energy (in GeV) for Higgs mass of 50MeV
    energy.append(np.random.normal(e, 0.107*np.sqrt(e))) #Energy of electron with a random deviation in range 0.107*np.sqrt(e)

plt.clf() #Close previous plot canvas                                        
plt.figure() #Open a new canvas                                                               
plt.hist(energy,bins = 100,color="blue", edgecolor="black") #Distribution of electron energy                     
plt.xlabel("Energy [Gev]")
plt.ylabel("Entries")
plt.annotate("mH = 50 MeV",xy=(0.025, 0.8), xycoords='axes fraction')
plt.savefig("energy_pb3_50MeV_with.pdf")
plt.show()

energy = [] #Reinitialisation of the list
for i in range(10000): #For 10000 initial events
    e = e_energy(10)/1e3 #List of electron energy (in GeV) for Higgs mass of 10MeV          
    energy.append(np.random.normal(e, 0.107*np.sqrt(e))) #Energy of electron with a random deviation in range 0.107*np.sqrt(e)

plt.clf() #Close previous plot canvas                                                                    
plt.figure() #Open a new canvas                                                                                       
plt.hist(energy,bins = 100,color="blue", edgecolor="black") #Distribution of electron energy                              
plt.xlabel("Energy [Gev]")
plt.ylabel("Entries")
plt.annotate("mH = 10 MeV",xy=(0.025, 0.8), xycoords='axes fraction')
plt.savefig("energy_pb3_10MeV_with.pdf")
plt.show()


energy = [] #Reinitialisation of the list
for i in range(10000): #For 10000 initial events
    e = e_energy(2)/1e3 #List of electron energy (in GeV) for Higgs mass of 2MeV
    energy.append(np.random.normal(e, 0.107*np.sqrt(e))) #Energy of electron with a random deviation in range 0.107*np.sqrt(e)

plt.clf() #Close previous plot canvas                       
plt.figure() #Open a new canvas                                                         
plt.hist(energy,bins = 100,color="blue", edgecolor="black") #Distribution of electron energy                           
plt.xlabel("Energy [Gev]")
plt.ylabel("Entries")
plt.annotate("mH = 2 MeV",xy=(0.025, 0.8), xycoords='axes fraction')
plt.savefig("energy_pb3_2MeV_with.pdf")
plt.show()
