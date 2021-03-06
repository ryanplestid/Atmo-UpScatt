'''
Module made to reweight a Monte Carlo integration for the HNL dipole portal with a new
    HNL mass and dipole coupling.  It requires the filename from a previous Monte Carlo integration
    to run.

Functions:
    Update_Fluxes: Changes the flavor dependent fluxes of an event to account for oscillations
    ReIntegrate: Recalculates the expected rate of visible events for a specified HNL mass
        and dipole coupling, given the filename from a previously run Monte Carlo with 
        similar parameters.
        
'''

import numpy as np
from numpy import sin, cos, pi
from numpy import random as rand 
import scipy as sp
from scipy import special as spc
import matplotlib as mpl #Package to help with plotting
from matplotlib import pyplot as plt
plt.style.use(['science','ieee'])

from matplotlib import ticker, cm
import pickle

#Import modules that we made
import dipoleModel
import atmoNuIntensity
import earthComp
import DetectorModule
import oscillations
import MainIntegration

def Update_Fluxes(filename,threshold):
    '''
    Recalculates the flux by taking oscillations into effect. Changes the event
        objects and alters the original file
    args:
        filename: string with the name of the file containing relevent information
        threshold: event's contribution to the total rate compared to average, below
                    which we do not update it.

    '''
    Sim_Dict = pickle.load(open(filename,"rb"))
    Meta_Data = Sim_Dict["MetaData"]
    Event_list = Sim_Dict["EventData"]
    
    #Don't perform oscillations if oscillations have already been done
    if Meta_Data['Oscillations'] == True:
        return(0)
    
    new_fluxes = {'E':[], 'EBar':[],
                  'Mu':[], 'MuBar':[],
                  'Tau':[], 'TauBar':[]}
    
    
    Rate = Meta_Data["Rate"]
    #min_dR = (Rate/N_Events) * threshold
    ###
    rates_list = np.zeros(len(Event_list))
    index = 0
    for event in Event_list:
        rates_list[index] = event.dR
        index += 1
    rates_list.sort()
    cummulative = 0
    min_index = -1
    while cummulative < Rate*threshold:
        min_index += 1
        cummulative += rates_list[min_index]
        
    min_dR = rates_list[min_index]
        
    ###
    trial = 0
    for event in Event_list:
        trial += 1
        
        if trial % 5000 == 0:
            print(trial)
            
        if event.dR < min_dR:
            new_fluxes['E'].append(event.NuEFlux)
            new_fluxes['Mu'].append(event.NuMuFlux)
            new_fluxes['Tau'].append(event.NuTauFlux)
            new_fluxes['EBar'].append(event.NuEbarFlux)
            new_fluxes['MuBar'].append(event.NuMubarFlux)
            new_fluxes['TauBar'].append(event.NuTaubarFlux)
            continue
        #print('did oscillate')
        X = event.Interact_Pos  #location of interaction in R_Earth = 1
        W = event.Entry_Pos    #location of production in R_Earth = 1

        Energy = event.nu_Energy     #Energy in GeV
        
        #print(r_W[0]**2 + r_W[1]**2 + r_W[2]**2)
        
        r_distance, n_e = earthComp.gen_1d_ne(W,X) #distance traveled (units of R_Earth) 
                                            #and number density of electrons (1/cm^3)
        #km_distance = r_distance * 6378.1
        probs = oscillations.getProbs(r_distance,n_e,Energy,anti = False)
        anti_probs = oscillations.getProbs(r_distance, n_e,Energy, anti = True)
        
        new_fluxes['E'].append(probs['e->e'] * event.NuEFlux 
                                  + probs['mu->e'] * event.NuMuFlux
                                  + probs['tau->e'] * event.NuTauFlux)
        
        new_fluxes['Mu'].append(probs['e->mu'] * event.NuEFlux 
                                  + probs['mu->mu'] * event.NuMuFlux
                                  + probs['tau->mu'] * event.NuTauFlux)
        
        new_fluxes['Tau'].append(probs['e->tau'] * event.NuEFlux 
                                  + probs['mu->tau'] * event.NuMuFlux
                                  + probs['tau->tau'] * event.NuTauFlux)
        
        new_fluxes['EBar'].append(anti_probs['e->e'] * event.NuEbarFlux 
                                  + anti_probs['mu->e'] * event.NuMubarFlux
                                  + anti_probs['tau->e'] * event.NuTaubarFlux)
        
        new_fluxes['MuBar'].append(anti_probs['e->mu'] * event.NuEbarFlux 
                                  + anti_probs['mu->mu'] * event.NuMubarFlux
                                  + anti_probs['tau->mu'] * event.NuTaubarFlux)
        
        new_fluxes['TauBar'].append(anti_probs['e->tau'] * event.NuEbarFlux 
                                  + anti_probs['mu->tau'] * event.NuMubarFlux
                                  + anti_probs['tau->tau'] * event.NuTaubarFlux)
        
    index = 0
    for event in Event_list:
        event.NuEFlux = new_fluxes['E'][index]
        event.NuMuFlux = new_fluxes['Mu'][index]
        event.NuTauFlux = new_fluxes['Tau'][index]
        event.NuEbarFlux = new_fluxes['EBar'][index]
        event.NuMubarFlux = new_fluxes['MuBar'][index]
        event.NuTaubarFlux = new_fluxes['TauBar'][index]
        index += 1
    
    Meta_Data['Oscillations'] = True
    
    Sim_Dict = {"MetaData":Meta_Data,"EventData":Event_list}
    with open(filename,'wb') as events_file:
        pickle.dump(Sim_Dict, events_file)
        
        
        
                                                

def ReIntegrate(filename,d,m_N,alpha_decay,V_det = None, flavors = ['E','EBar','Mu','MuBar','Tau','TauBar']):
    '''
    Re-integrates to find the rate given a file with meta data and event objects
    
    args:
        filename: Name of the file that contains the necessary information (str)
        d: Dipole coupling constant (MeV^-1)
        m_N: HNL mass (GeV)
        V_det: Volume of the detector in cm^3 (float).  If left as None, it will use the volume
            given in the file.
        flavors: Flavors of neutrinos for which we are interested in the interactions
    
    returns:
        tot_integral: Rate of sub-GeV HNL events in the detector in s^{-1} (float)
        rate_error: Uncertainty in the rate in s^{-1} (float)
        cos_phi_midpoints: Midpoints of the bins in cos(phi_det) (array)
        angular_rates: rate in each bin of cos(phi_det) in s^{-1} (array)
        E_gamma_midpoints: Midpoints of the bins in photon energy in GeV (array)
        E_gamma_rates: Rate in each bin of photon energy in s^{-1} (array)
        high_E_rate: Rate of multi-GeV HNL events in the detector in s^{-1} (float)

    '''
    Sim_Dict = pickle.load(open(filename,"rb"))
    Meta_Data = Sim_Dict["MetaData"]
    Event_list = Sim_Dict["EventData"]
    
    Y = Meta_Data['Detector Location (r_Earth=1)']
    R_Earth = 1 # R_Earth = 1 #
    R_Earth_cm = 6378.1 * 1000* 100    #Radius of the Earth (cm)
    
    Energy_range = Meta_Data['Energy Limits'][1] - Meta_Data['Energy Limits'][0] #GeV
    Theta_min,Theta_max = Meta_Data['Theta Limits'][0], Meta_Data['Theta Limits'][1]
    Cos_Theta_range = cos(Theta_min) - cos(Theta_max)
    Earth_Volume = (4*pi/3)*R_Earth**3 #cm^3
    
    if V_det == None:
        V_det = Meta_Data["Detector Volume (cm^3)"]
    
    A_perp = (V_det)**(2/3) #cm^2
    l_det = (V_det)**(1/3) / R_Earth_cm # units of R_Earth = 1
    Energies = np.zeros(len(Event_list))
    dRs = np.zeros(len(Event_list))
    N_Fluxes = np.zeros(len(Event_list))
    X_vect_vals = np.zeros((len(Event_list),3))
    N_sigma_cos_Thetas = np.zeros(len(Event_list))
    cos_Thetas = np.zeros(len(Event_list))
    
    
    trial = 0
    for event in Event_list:
        Energies[trial] = event.N_Energy
        X_vect_vals[trial,:] = event.Interact_Pos
        
        N_lambda = dipoleModel.decay_length(d,m_N, event.N_Energy)
        
        if flavors == ['E','EBar','Mu','MuBar','Tau','TauBar']:
            N_lambda = N_lambda/3
            
        P_dec_A_Perp = (np.exp(-event.Dist_to_det/N_lambda) * (1-np.exp(-l_det/N_lambda))
                        *A_perp)
        
        #Find the total flux for desired flavors
        tot_flux = 0
        for flavor in flavors:
            if flavor == 'E':
                tot_flux += event.NuEFlux
            elif flavor == 'EBar':
                tot_flux += event.NuEbarFlux
            elif flavor == 'Mu':
                tot_flux += event.NuMuFlux
            elif flavor == 'MuBar':
                tot_flux += event.NuMubarFlux
            elif flavor == 'Tau':
                tot_flux += event.NuTauFlux
            elif flavor == 'TauBar':
                tot_flux += event.NuTaubarFlux
            else:
                print(flavor)
        
        weights = event.E_weight * event.V_weight * event.Theta_weight
        tot_delta = R_Earth_cm* Energy_range * Cos_Theta_range * Earth_Volume * weights / len(Event_list)
        N_d_sigma_d_cos_Theta = dipoleModel.N_Cross_Sec_from_radii(d,m_N,event.nu_Energy,
                                                                         event.cos_Theta,event.r)
        
        N_sigma_cos_Thetas[trial] = N_d_sigma_d_cos_Theta
        cos_Thetas[trial] = event.cos_Theta

        dRs[trial] = (N_d_sigma_d_cos_Theta *4*pi*tot_flux 
                         * (P_dec_A_Perp)/(4*pi*event.Dist_to_det**2) * tot_delta)
        
        N_Fluxes[trial] = (N_d_sigma_d_cos_Theta *4*pi*tot_flux 
                         * np.exp(-event.Dist_to_det/N_lambda) /(4*pi*event.Dist_to_det**2) * tot_delta)
        
        trial += 1
    
    cos_phi_bounds = np.linspace(-1,1,11)
    cos_phi_midpoints = (cos_phi_bounds[:10] + cos_phi_bounds[1:])/2
    angular_rates = np.zeros(len(cos_phi_midpoints))
    
    E_gamma_bounds = np.linspace(0.03, 1.33 ,11)
    E_gamma_midpoints = (E_gamma_bounds[:10] + E_gamma_bounds[1:])/2
    E_gamma_rates = np.zeros(len(E_gamma_midpoints))
    
    Used_dRs = np.array([])
    
    tot_integral = 0
    high_E_rate = 0
    
    for i in range(10):
    
        cos_zeta_primes = DetectorModule.Calc_cos_zeta_prime(len(Event_list),alpha_decay)
        zetas, E_gammas = DetectorModule.Calc_Zetas_and_Energies(cos_zeta_primes, Energies, m_N)
        cos_phi_dets = DetectorModule.Calc_cos_phi_det(Y, X_vect_vals, zetas)
        
        for e_index in range(len(cos_phi_dets)):
            cos_phi = cos_phi_dets[e_index]
            E_gamma = E_gammas[e_index]
            dR = dRs[e_index]/10
            if E_gammas[e_index] > 0.030 and E_gammas[e_index] < 1.33:
                tot_integral += dR
                Used_dRs = np.append(Used_dRs,dR)
                
                for phi_index in range(len(angular_rates)):
                    if cos_phi_bounds[phi_index] < cos_phi and cos_phi < cos_phi_bounds[phi_index + 1]:
                        angular_rates[phi_index] += dR
                
                for energy_index in range(len(E_gamma_rates)):
                    if E_gamma_bounds[energy_index] < E_gamma and E_gamma < E_gamma_bounds[energy_index+1]:
                        E_gamma_rates[energy_index] += dR
                        
            if E_gamma > 1.33:
                high_E_rate += dR
    print('Rate',tot_integral)
    
    rate_error = np.std(Used_dRs)*np.sqrt(len(Used_dRs))
    
    print('Rate Error', rate_error)
    
    return(tot_integral,rate_error,cos_phi_midpoints, angular_rates, E_gamma_midpoints,E_gamma_rates,high_E_rate)


