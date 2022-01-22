'''
Module made to reweight a Monte Carlo integration for the HNL mass-mixing portal with a new
    HNL mass and coupling strength.  It requires the filename from a previous Monte Carlo 
    integration to run.
'''

import numpy as np
from numpy import sin, cos, pi
from numpy import random as rand 
import scipy as sp
from scipy import special as spc
import matplotlib as mpl #Package to help with plotting
from matplotlib import pyplot as plt
from matplotlib import ticker, cm
import pickle
plt.style.use(['science','ieee'])

#Import modules that we made
#import SampleEvents
import Mass_Mixing_Model
import atmoNuIntensity
import earthComp
import DetectorModule
import oscillations
import MassMixingMainIntegration
from MassMixingMainIntegration import MassMixMonteCarlo
import MassMixingDetectorModule
from reweightModule import Update_Fluxes

def ReIntegrate(filename,U,m_N,scattering_channel, V_det = None, flavors = ['E','EBar','Mu','MuBar','Tau','TauBar'], Flav = 'e'):
    '''
    Re-integrates to find the rate given a file with meta data and event objects
    
    args:
        filename: Name of the file that contains the necessary information (str)
        U = Mass Mixing coupling
        m_N = HNL mass (GeV)
        flavors: Flavors of neutrinos for which we are interested in the interactions
        Flav: Flavor of the HNL coupling, dictates the type of interactions that can happen
    
    returns:
        tot_integral: Rate of sub-GeV HNL events in the detector in s^{-1} (float)
        rate_error: Uncertainty in the rate in s^{-1} (float)
        E_ee_midpoints: Midpoints of the bins in lepton pair energy in GeV (array)
        E_ee_rates: Rate in each bin of lepton pair energy in s^{-1} (array)

    '''
    #Perform Oscillations
    Update_Fluxes(filename, 0.1)
    
    #Load Data
    Sim_Dict = pickle.load(open(filename,"rb"))
    Meta_Data = Sim_Dict["MetaData"]
    Event_list = Sim_Dict["EventData"]
    
    Y = Meta_Data['Detector Location (r_Earth=1)']
    R_Earth = 1 # R_Earth = 1 #
    R_Earth_cm = 6378.1 * 1000* 100    #Radius of the Earth (cm)
    
    Energy_range = Meta_Data['Energy Limits'][1] - Meta_Data['Energy Limits'][0] #GeV
    Theta_min= Meta_Data['Theta min']
    Theta_max = Meta_Data['Theta max']
    
    Earth_Volume = (4*pi/3)*R_Earth**3 #cm^3
    
    if V_det == None:
        V_det = Meta_Data["Detector Volume (cm^3)"]
    
    A_perp = (V_det)**(2/3) #cm^2
    l_det = (V_det)**(1/3) / R_Earth_cm # units of R_Earth = 1
    E_nus = np.zeros(len(Event_list))
    E_HNLs = np.zeros(len(Event_list))
    dRs = np.zeros(len(Event_list))
    N_Fluxes = np.zeros(len(Event_list))
    X_vect_vals = np.zeros((len(Event_list),3))
    N_sigma_cos_Thetas = np.zeros(len(Event_list))
    cos_Thetas = np.zeros(len(Event_list))
    
    tot_integral = 0
    trial = 0
    for event in Event_list:
        Cos_Theta_range = cos(Theta_min) - cos(Theta_max[trial])
        E_nus[trial] = event.nu_Energy
        E_HNLs[trial] = event.N_Energy
        X_vect_vals[trial,:] = event.Interact_Pos
        
        #N_lambda = dipoleModel.decay_length(d,m_N, event.N_Energy)
        N_lambda = Mass_Mixing_Model.decay_length(Flav, U, m_N, np.array([E_HNLs[trial]]))
            
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
        N_d_sigma_d_cos_Theta = Mass_Mixing_Model.N_Cross_Sec_from_radii(U,m_N,[event.nu_Energy],
                                                                         [event.cos_Theta],[event.r],
                                                                         scattering_channel)
        
        N_d_sigma_d_cos_Theta = float(N_d_sigma_d_cos_Theta)
        N_sigma_cos_Thetas[trial] = N_d_sigma_d_cos_Theta
        cos_Thetas[trial] = event.cos_Theta
        
        visible_frac = Mass_Mixing_Model.Gamma_partial(Flav, m_N, U, final_state = 'nu e e')\
                /Mass_Mixing_Model.Gamma_tot(Flav,m_N,U)
        
        dRs[trial] = (N_d_sigma_d_cos_Theta *4*pi*tot_flux 
                         * (P_dec_A_Perp)/(4*pi*event.Dist_to_det**2) * tot_delta 
                         * visible_frac)
        

        N_Fluxes[trial] = (N_d_sigma_d_cos_Theta *4*pi*tot_flux 
                         * np.exp(-event.Dist_to_det/N_lambda) /(4*pi*event.Dist_to_det**2) * tot_delta)
        
        trial += 1

    
    #find energies for the e+ e- pair
    E_ee_bounds = np.linspace(0.03, 1.33 ,11)
    E_ee_midpoints = (E_ee_bounds[:10] + E_ee_bounds[1:])/2
    E_ee_rates = np.zeros(len(E_ee_midpoints))
    rate = 0
    high_E_rate = 0
    
    #Find energy of the lepton pair in the lab frame
    decay_trials = 10
    for i in range(decay_trials):
        E_decay_nu_lab = MassMixingDetectorModule.Sample_Decay_E_nu_Lab(m_N, E_HNLs, 'Vector')
        E_ees = E_HNLs - E_decay_nu_lab
        
        for event_index in range(len(E_ees)):
            E_ee = E_ees[event_index]
            dR = dRs[event_index] / decay_trials
            #See if lepton pair energy makes cut
            if E_ee > 0.030 and E_ee < 1.33:
                rate += dR
                
                for energy_index in range(len(E_ee_rates)):
                    if E_ee_bounds[energy_index]<E_ee and E_ee < E_ee_bounds[energy_index+1]:
                        E_ee_rates[energy_index] += dR
            if E_ee > 1.33:
                high_E_rate += dR
        
    
    
    rate_error = np.std(dRs)*np.sqrt(len(dRs))
    
    print('% Error', (rate_error/rate)*100)
    
    return(rate,rate_error, E_ee_midpoints, E_ee_rates)
