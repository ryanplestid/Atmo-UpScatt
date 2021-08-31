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
#import SampleEvents
import dipoleModel
import Incoherent_Module
import atmoNuIntensity
import earthComp
#import DetectorModule
import oscillations
import MainIntegration
#from MainIntegration import filename_list, d_vals, m_N_vals, num_Events

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
    
    new_fluxes = {'E':[], 'EBar':[],
                  'Mu':[], 'MuBar':[],
                  'Tau':[], 'TauBar':[]}
    
    N_Events = len(Event_list)
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
        
        if trial % 100000 == 0:
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
        
    Sim_Dict = {"MetaData":Meta_Data,"EventData":Event_list}
    with open(filename,'wb') as events_file:
        pickle.dump(Sim_Dict, events_file)
        
        
        
                                                

def ReIntegrate(filename,d,m_N, flavors = ['E','EBar','Mu','MuBar','Tau','TauBar']):
    '''
    Re-integrates to find the rate given a file with meta data and event objects
    
    args:
        filename: Name of the file that contains the necessary information (str)
        d: Dipole coupling constant (MeV^-1)
        m_N = HNL mass (GeV)
        flavors: Flavors of neutrinos for which we are interested in the interactions
    
    returns:
        tot_integral: Rate of decays in the detector (s^-1)

    '''
    Sim_Dict = pickle.load(open(filename,"rb"))
    Meta_Data = Sim_Dict["MetaData"]
    Event_list = Sim_Dict["EventData"]
    R_Earth = 1 # R_Earth = 1 #
    R_Earth_cm = 6378.1 * 1000* 100    #Radius of the Earth (cm)
    
    Energy_range = Meta_Data['Energy Limits'][1] - Meta_Data['Energy Limits'][0] #GeV
    Theta_min,Theta_max = Meta_Data['Theta Limits'][0], Meta_Data['Theta Limits'][1]
    Cos_Theta_range = cos(Theta_min) - cos(Theta_max)
    Earth_Volume = (4*pi/3)*R_Earth**3 #cm^3
    V_det = Meta_Data["Detector Volume (cm^3)"]
    
    A_perp = (V_det)**(2/3) #cm^2
    l_det = (V_det)**(1/3) / R_Earth_cm # units of R_Earth = 1
    
    tot_integral = 0
    trial = 0
    for event in Event_list:
        trial += 1
        N_lambda = dipoleModel.decay_length(d,m_N, event.N_Energy)
            
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
        N_d_sigma_d_cos_Theta = Incoherent_Module.N_Cross_Sec_from_radii(d,m_N,event.nu_Energy,
                                                                         event.cos_Theta,event.r)
        tot_integral += (N_d_sigma_d_cos_Theta *4*pi*tot_flux 
                         * (P_dec_A_Perp)/(4*pi*event.Dist_to_det**2) * tot_delta)
        
    
    print('Rate',tot_integral)
    return(tot_integral)
