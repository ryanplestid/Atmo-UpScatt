def main():
    print("Hello World")
    return 0

import nuflux #Package to accurately determine the flux of neutrinos at different energies and angles
import numpy as np #Package for array functions
from numpy import pi as pi
import scipy as sp
from scipy import special as spc
from numpy import random as rand
from numpy import sin, cos

def Sample_Neutrino_Energies(num_Events, E_min, E_max, power_law):
    '''
    This function samples neutrino energies according to a power law
    
    args:
        num_Events: number of energies we wish to sample (int)
        E_min: minimum Energy of our distribution in GeV (float)
        E_max: maximum Energy of our distriubtion in GeV (float)
        power_law: our distibution follows E^{-power_law} (float)
    
    returns:
        Energies: array of length num_Events with the sampled energies (floats)
    '''
    kappa = (1 - power_law)/ (E_max**(1-power_law) - E_min**(1-power_law)) #Constant used in sampling
    rand_chi = rand.rand(num_Events)  #array of random numbers between 0 and 1
    
    first_terms = ((1-power_law)/kappa) * rand_chi
    second_term = E_min**(1-power_law)
    
    Energies = (first_terms+second_term)**(1/(1-power_law))
    
    return(Energies)

def Calculate_dE(num_Events, Energies, E_min, E_max, power_law):
    '''
    This function returns the properly weighted values of dE
    
    args:
        num_Events: number of events in our integral(int)
        Energies: Neutrino energies of our events (array of floats)
        E_min: minimum Energy of our distribution in GeV (float)
        E_max: maximum Energy of our distriubtion in GeV (float)
        power_law: our distibution follows E^{-power_law} (float)
    
    returns:
        dEs: weighted differentials for calculating the integral
            (array of floats, same size as Energies)
    '''
    num = (1-power_law)*(E_max - E_min)
    den = E_max**(1-power_law) - E_min**(1-power_law)
    E_char = (num/den)**(1/power_law) #Characteristic Energy
    
    dEs = (E_max - E_min)/(num_Events**(1/3)) * (Energies/E_char)**power_law
    
    return(dEs)

def Calc_cos_zeniths(interaction_locs, entry_locs):
    '''
    Calculates the zenith angles for events given the interaction
    location and entry location
    
    args:
        interaction_locs: n-by-3 array for the cartesian interaction positions of
                n events in cm (floats)
        entry_locs: n-by-3 array for the cartesian entry positions of
                n events in cm (floats)
                
    returns
        cos_zeniths: array of n elements of the cosines of the
                incoming neutrinos
    '''
    R_Earth = 6378.1 * 1000* 100    #Radius of the Earth (cm)
    
    #Calculate the dot products
    dot_prods = (interaction_locs[:,0]*entry_locs[:,0] + interaction_locs[:,1]*entry_locs[:,1]
                 + interaction_locs[:,2]*entry_locs[:,2])
    
    #Calculate the difference vector
    diff_vecs = np.zeros((len(interaction_locs),3))
    diff_vecs[:,0] = interaction_locs[:,0] - entry_locs[:,0]
    diff_vecs[:,1] = interaction_locs[:,1] - entry_locs[:,1]
    diff_vecs[:,2] = interaction_locs[:,2] - entry_locs[:,2]
    diff_mags = np.sqrt(diff_vecs[:,0]**2 + diff_vecs[:,1]**2 + diff_vecs[:,2]**2)
    
    #Calculate the cosines of the zenith angles
    cos_zeniths = (R_Earth**2 - dot_prods)/(R_Earth*diff_mags)
    
    return(cos_zeniths)

def calc_fluxes(Energies,cos_zeniths,flux_name,nu_flavors = ['E','EBar','Mu','MuBar','Tau','TauBar']):
    '''
    Calculate the flux of specified neutrino components
    
    args:
        Energies: array of initial neutrino energies in GeV (floats)
        cos_zeniths: array of cosines of zenith angles relative to
                    the surface of Earth for incoming neutrinos (floats)
        flux_name: Name of the flux to use for NuFlux (string)
        nu_flavors: List of strings specifying the desired neutrino flavors
                    If no input, then all types will be calculated.
    
    returns:
        fluxes: array of the neutrino fluxes at the specified energy and
            zenith angle in GeV^-1 cm^-2 sr^-1 s^-1 (floats)
    '''
    #All Units in NuFlux are (GeV^-1 cm^-2 sr^-1 s^-1)
    flux = nuflux.makeFlux(flux_name) #Flux to be used for calculations
    
    #Initialize the different flavors of neutrinos
    nu_e_type, nu_e_bar_type = nuflux.NuE, nuflux.NuEBar
    nu_mu_type, nu_mu_bar_type = nuflux.NuMu, nuflux.NuMuBar
    nu_tau_type, nu_tau_bar_type = nuflux.NuTau, nuflux.NuTauBar
    
    nu_dict = dict({'E':nu_e_type, 'EBar':nu_e_bar_type,
                   'Mu':nu_mu_type, 'MuBar':nu_mu_bar_type,
                   'Tau':nu_tau_type, 'TauBar':nu_tau_bar_type})
    
    #Initialize fluxes
    fluxes = np.zeros(len(Energies))
    
    #Added fluxes for each desired flavor.
    for flavor in nu_flavors:
        fluxes = fluxes + flux.getFlux(nu_dict[flavor],Energies,cos_zeniths)
    
    return(fluxes)


if __name__ == "__main__":
    main()
