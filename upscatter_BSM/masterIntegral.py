def main():
    print("Hello World")
    return 0

import numpy as np
from numpy import sin, cos, pi
from numpy import random as rand 
import scipy as sp
from scipy import special as spc
import matplotlib #Package to help with plotting
from matplotlib import pyplot as plt
from matplotlib import ticker, cm
import pickle

#Import modules that we made
import SampleEvents
import dipoleModel
import atmoNuIntensity
import earthComp
import DetectorModule
import Incoherent_Module

#Make Event Class
class Event :
    def __init__(self, E_nu, E_N, X,W, cos_zenith_angle):
        self.nu_Energy = E_nu #Neutrino Energy GeV
        self.N_Energy = E_N #HNL Energy GeV
        self.Interact_Pos = X # Position of interaction (cm)
        self.Entry_Pos = W #Position of neutrino entry (cm)
        self.cos_zen = cos_zenith_angle #cosine of zenith angle for neutrino entry

#Flavors for the different neutrinos        
flavors = ['E','EBar','Mu','MuBar','Tau','TauBar']

#Parameters
E_min,E_max = 0.1, 1e4 #min/max Neutrino Energy, GeV
power_law = 2 #Guess at energy dependence of atm flux
flux_name= 'H3a_SIBYLL23C'

R_Earth = 1  # use units where R_earth = 1
R_Earth_cm = 6378.1 * 1000* 100    #Radius of the Earth (cm)
c = 10 #R_max = c*lambda
Y = np.array([0,0,0.9999*R_Earth])     #Location of our detector
V_det = 32000 * (1e6)         #Volume of detector in cm^3

R_min = (V_det)**(1/3) / R_Earth_cm #minimum distance to detector in R_Earth = 1

A_perp = (V_det)**(2/3) #cm^2
l_det = (V_det)**(1/3) / R_Earth_cm # units of R_Earth = 1

epsilon = 1e-7 #1 - cos(Theta_min)
Theta_max = pi

def MonteCarlo(m_N_vals,d_vals,num_Events):
    
    '''
    args:
        m_N_vals: array of HNL masses (GeV)
        d_vals: array of dipole couplings (MeV^-1)
        num_Events: int for the number of events in the Monte Carlo
    returns:
        filename: string of the file with the important information
        Decays: 2D array for the rate of decays (1/s)
    '''
    
    Decays = np.zeros((len(d_vals),len(m_N_vals)))
    
    
    alpha_decay = 0 #Used for determining if the HNL is Dirac or Majoranna
    m_nuc = 0.94 #nucleon mass GeV
    
    filename_list = []
    #Sample events
    d_index = 0
    for d in d_vals:
        
        m_N_index = 0
        for m_N in m_N_vals:
            print(d*1e9,'*1e-9 (MeV^-1)')
            print(m_N, 'GeV')
            #Make a dictionary of the MetaData for the simulation
            Sim_MetaData = {'Flux Choice': flux_name,
                            'Earth Model': 'PREM',
                            'Energy Power Law':-power_law,
                            'Theta Limits': [np.arccos(1-epsilon), Theta_max],
                            'Cos Theta Power Law':-1,
                            'd (MeV^-1)': d,
                            'm_N (GeV)': m_N,
                            'Detector Location (r_Earth=1)':Y,
                            'Detector Volume (cm^3)':V_det}
            
            #Find Energy to allow scattering at all angles (GeV)
            if m_N < m_nuc:
                E_thresh = ((m_nuc * m_N**2) + m_N * (2* m_nuc**2 - m_N**2))/(2 * m_nuc**2 - 2* m_N**2)
            else:
                E_thresh = ((m_nuc * m_N**2) + m_N * (-2* m_nuc**2 + m_N**2))/(2 * m_nuc**2 - 2* m_N**2)
            print(E_thresh)
            
            E_min = max(0.1,E_thresh) #Threshold Energy for scattering off proton
        
            Energies = SampleEvents.Sample_Neutrino_Energies(num_Events,E_min,E_max,power_law)
            lambdas = dipoleModel.decay_length(d,m_N,Energies)
            R_maxs = c*lambdas + np.ones(len(lambdas)) * 1.5 * R_min
            
            X_vect_vals = SampleEvents.Sample_Interaction_Locations(num_Events, Y, R_maxs,R_min)
            Radii = np.sqrt(X_vect_vals[:,0]**2 +X_vect_vals[:,1]**2 + X_vect_vals[:,2]**2) #cm
            rs = Radii/R_Earth #normalized radii
            cos_Thetas = SampleEvents.Sample_cos_Theta(num_Events, epsilon, Theta_max)
    
            ## Find where each neutrino entered earth's crust
            W_vect_vals = SampleEvents.Sample_Neutrino_Entry_Position(X_vect_vals,Y,cos_Thetas)
            
            #Calculate Flux
            cos_zeniths = atmoNuIntensity.Calc_cos_zeniths(X_vect_vals,W_vect_vals)
            flux = atmoNuIntensity.calc_fluxes(Energies,cos_zeniths, flux_name) #Flux in GeV^-1 cm^-2 sr^-1 s^-1
            
            #Calculate cross section x number density
            N_d_sigma_d_cos_Thetas = Incoherent_Module.N_Cross_Sec_from_radii(d,m_N,Energies,cos_Thetas,rs)
            
            #Prob to decay in detector * Perpendicular Area
            Y_minus_X_mag = np.sqrt((Y[0] - X_vect_vals[:,0])**2 + (Y[1] - X_vect_vals[:,1])**2 + (Y[2] - X_vect_vals[:,2])**2)
            print(min(Y_minus_X_mag)/l_det, 'min distance/l_det')
            P_decs_A_Perp = (np.exp(-Y_minus_X_mag/lambdas) *(1 - np.exp(-l_det/lambdas))
                             * A_perp)
            
            #Calculate weights for events
            E_Range = E_max - E_min
            cos_Theta_range = (1-epsilon) - cos(Theta_max)
            Volume = 4*pi/3 * R_Earth ** 3
    
            w_E = SampleEvents.weight_Energy(Energies, E_min, E_max, power_law)
            w_V = SampleEvents.weight_positions(Y,R_maxs, R_min)
            w_Theta = SampleEvents.weight_cos_Theta(cos_Thetas,epsilon, Theta_max)
            tot_weight = (w_E * w_V * w_Theta)
            
            tot_delta = R_Earth_cm * E_Range * cos_Theta_range * Volume * tot_weight/num_Events
            
            #Calculate how much each event contributes to the rate
            
            dR = N_d_sigma_d_cos_Thetas * 4 * pi * flux * (P_decs_A_Perp)/(4*pi * Y_minus_X_mag**2) * tot_delta
            Rate = sum(dR)
            Decays[d_index, m_N_index] = Rate
            #print('Number of Events', num_Events)
            #print('Rate', Rate)
            print('decays in 10 years', Rate * 86400 * 365 *10)
            print('')
            
            #Calculate the flavor specific fluxes
            flavor_fluxes = {}
            for flavor in flavors:
                flavor_fluxes[flavor] = atmoNuIntensity.calc_fluxes(Energies,cos_zeniths,
                                          flux_name,nu_flavors =[flavor])
    
            #Create a filename to save these events
            filename = "mN_"+str(m_N) +"_d_"+str(d)+".events"
            #Make a list of events
            event_list = []
            for i in range(len(Energies)):
                E_nu,E_N = Energies[i], Energies[i]
                interact_pos = X_vect_vals[i,:]
                entry_pos = W_vect_vals[i,:]
                cos_zen = cos_zeniths[i]
                event = Event(E_nu, E_N, interact_pos, entry_pos,cos_zen)
                
                event.r = rs[i]
                event.cos_Theta = cos_Thetas[i]
                event.Dist_to_det = Y_minus_X_mag[i]
                event.N_lambda = lambdas[i]
                event.N_diff_cross_sec = N_d_sigma_d_cos_Thetas[i]
                
                event.NuEFlux = flavor_fluxes['E'][i]
                event.NuMuFlux = flavor_fluxes['Mu'][i]
                event.NuTauFlux = flavor_fluxes['Tau'][i]
                
                event.NuEbarFlux = flavor_fluxes['EBar'][i]
                event.NuMubarFlux = flavor_fluxes['MuBar'][i]
                event.NuTaubarFlux = flavor_fluxes['TauBar'][i]
                
                event.E_weight = w_E[i]
                event.V_weight = w_V[i]
                event.Theta_weight = w_Theta[i]
                
                event.dR = dR[i]
                
                event_list.append(event)
            Sim_MetaData['Energy Limits']=[E_min,E_max]
            Sim_MetaData['Rate'] = Rate
            
            Sim_Dictionary = {'MetaData':Sim_MetaData,
                              'EventData': event_list}
            with open(filename,'wb') as events_file:
                pickle.dump(Sim_Dictionary, events_file)
            
            filename_list.append(filename)
            #print(Sim_Dictionary['MetaData']['m_N (GeV)'])
            
            m_N_index += 1
        
        d_index += 1
    return(filename, Decays)


'''
figk, ax = plt.subplots(1,1)
levels = np.logspace(-6,8,8)
cp = ax.contourf(m_N_vals, d_vals, Decays * 86400 * 365 * 10, levels,locator = ticker.LogLocator(), cmap = cm.PuBu_r)
figk.colorbar(cp)
plt.ylabel('d ($MeV^{-1}$)', fontsize = 14)
plt.yscale('log')
plt.xlabel('$M_N$ (GeV)', fontsize = 14)
plt.xscale('log')
plt.suptitle('Total Decays over 10 years', fontsize = 16)
plt.suptitle('Monte Carlo Total HNL Decays', fontsize = 16)
plt.title(str(num_Events) + ' events, Super-K Volume, No Double Bang, Full Cross Sec')
'''
'''
#Calculate what is seen by the detector
cos_zeta_primes = DetectorModule.Calc_cos_zeta_prime(num_Events,alpha_decay)
zetas, E_gammas = DetectorModule.Calc_Zetas_and_Energies(cos_zeta_primes, Energies, m_N)
cos_phi_dets = DetectorModule.Calc_cos_phi_det(Y, X_vect_vals, zetas)
E_midpoints, cos_midpoints, rates = DetectorModule.Rate_In_Each_Bin(0,1.3, 5, 10,E_gammas, cos_phi_dets, dR)
figk, ax = plt.subplots(1,1,figsize = (8,6))
#levels = np.logspace(-3,3,20)
cp = ax.contourf(cos_midpoints,E_midpoints,rates*86400*365*10)
figk.colorbar(cp)
plt.ylabel('Energy (GeV)')
plt.xlabel('$\cos(\\theta_z)$')
plt.ylim([min(E_midpoints),0.7])
#plt.suptitle('Total events in each bin for 10 years',fontsize = 16)
#plt.title('$m_N$ ='+str(m_N)+"GeV, d = "+str(d)+"$MeV^{-1}$",fontsize = 14)
'''


if __name__ == "__main__":
    main()
