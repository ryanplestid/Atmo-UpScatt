def main():
    print("Hello World")
    return 0

#Initialization
import numpy as np #Package for array functions
import formFactorFit
from numpy import random as rand
from numpy import sin, cos

#Function to calculate the decay length of
#   the neutral lepton
def decay_length(d,mn,En):
    '''
    Find the characteristic decay length of the lepton
    
    args:
        d: dipole coupling constant in MeV^-1 (float)
        mn: mass of the lepton in GeV (float)
        En: Energy of the lepton in GeV (float or array of floats)
    
    returns:
        Lambda: Characteristic decay length in cm (float or array of floats, equal to the number of energies)
        
    action:
        Calculates characteristic decay length according to Plestid, 2021
    '''
    #Set decay length to 0 if the mass is greater than the energy
    if mn >= En:
        Lambda = 0
        return(Lambda)
    
    R_Earth = 6378.1 * 1000* 100    #Radius of the Earth (cm)
    mn_MeV = mn*1000  #Convert mass to MeV
    En_MeV = En*1000  #Convert energy to MeV
    Lambda = (R_Earth * (1.97e-9/d)**2 *(1/mn_MeV)**4 * (En_MeV/10) 
              * np.sqrt((1-mn**2/En**2)/0.99) )#Decay lengths (cm)
    
    return(Lambda)

#Function to calculate the fully coherent
#   scattering cross section
def d_sigma_d_cos_Theta_coherent(d,mn,En,cos_Theta,Zed):
    '''
    Determine the differential cross section for a coherent scattering at a specified angle
    
    args:
        d: dipole coupling constant in MeV^-1 (float)
        mn: mass of the lepton in GeV (float)
        En: Energy of the lepton in GeV (float or array of floats)
        cos_Theta: cosine of the scattering angle (float, same size as En)
        Zed: Atomic number (int)
        
    returns:
        d_sigma_d_cos_Theta: differential cross section in cm of the scattering
                            (float, same size as En)
    
    actions:
        Computes the differential cross section in terms of MeV^{-2}, then
        converts it to cm^2
    '''
    #Setting cross section to 0 if mass is larger than the energy
    if mn >= En:
        d_sigma_d_cos_Theta = 0
        return(d_sigma_d_cos_Theta)
    
    alpha = 1/137 #Fine structure constant (SI Units)
    En_MeV = En *1000  #Neutrino/Lepton Energies in MeV
    mn_MeV = mn*1000  #Lepton mass in MeV
    t = (2*En_MeV**2 - mn_MeV**2 - 2*En_MeV*np.sqrt(En_MeV**2 - mn_MeV**2) 
          * cos_Theta) #Transfered momentum^2 (MeV^2)

    leading_terms = (-2* np.sqrt(En_MeV**2 - mn_MeV**2) * Zed**2 * d**2 * alpha)/(En_MeV*t) #MeV^-4
    second_terms = (4*En_MeV**2 - mn_MeV**2 + mn_MeV**4/t) #MeV^2
    Inv_Mev_to_cm = (197.3) * 1e-13 # (MeV^-1 to fm) * (fm to cm)
    d_sigma_d_cos_Theta = Inv_Mev_to_cm**2 * -leading_terms*second_terms #cm^2
    
    return(d_sigma_d_cos_Theta)

#Function to calculate the scattering cross section
#   with form factors and multiple elements
def Full_d_sigma_d_cos_Theta(d, mn, En, cos_Theta, Zeds, R1s, Ss, fracs):
    '''
    Determine the differential cross section for a coherent and incoherent 
    scattering at a specified angle
    
    args:
        d: dipole coupling constant in MeV^-1 (float)
        mn: mass of the lepton in GeV (float)
        En: Energy of the lepton in GeV (float or array of floats)
        cos_Theta: cosine of the scattering angle (float, same size as En)
        Zeds: Atomic numbers (array of ints)
        R1s: Helm effective radii in fm (array of floats, same size as Zeds)
        Ss: Helm skin thicknesses in fm (array of floats, same size as Zeds)
        fracs: fractional number density of the nucleus in question, should sum to 1
                (array of floats, same size as Zeds)
        
    returns:
        d_sigma_d_cos_Theta: differential cross section in cm of the scattering
                            (float, same size as En)
    '''
    #Set differential cross section to 0 if the mass is greater than the energy
    if mn >= En:
        d_sigma_d_cos_Theta = 0
        return(d_sigma_d_cos_Theta)
    
    #Calculate the transfered momentum
    En_MeV = En *1000  #Neutrino/Lepton Energies in MeV
    mn_MeV = mn*1000  #Lepton mass in MeV
    t = (2*En_MeV**2 - mn_MeV**2 - 2*En_MeV*np.sqrt(En_MeV**2 - mn_MeV**2) 
          * cos_Theta) #Transfered momentum^2 (MeV^2)
    
    q = np.sqrt(t) #Transfered momentum (MeV)
    
    #Calculate the coherent cross sections for Z = 1
    d_sig_d_cos_coh = d_sigma_d_cos_Theta_coherent(d,mn,En,cos_Theta,1)
    
    #Initialize the cross section as 0
    d_sigma_d_cos_Theta = 0
    
    #Iterate through the nuclei
    for Zed_index in range(len(Zeds)):
        Zed = Zeds[Zed_index] #Atomic number
        R1 = R1s[Zed_index] #Effective nuclear radius
        s = Ss[Zed_index]  #Skin thickness
        frac = fracs[Zed_index]  #fractional number density of the nucleus
        
        FF2 = formFactorFit.Helm_FF2(q,R1,s) #Form factors^2 for the transferred momentum
        
        d_sigma_d_cos_Theta += frac * d_sig_d_cos_coh * Zed**2 * FF2
    
    return(d_sigma_d_cos_Theta)

#Function to sample scattering angles characteristic
#   of our dipole case
def Sample_cos_Theta(Num_Events, epsilon):
    '''
    Samples cosines of the scattering angles according to a 1/(1-cos(Theta)) distribution
    
    args:
        Num_Events: number of scattering angles that we wish to sample (int)
        epsilon: how close the max value of cos(Theta) can be to 1 (float)
    
    returns:
        cos_Thetas: sampled scattering angles (array of length Num_Events, floats)
    '''
    cos_theta_min = -1 #Minimum possible cos(Theta) value
    cos_theta_max = 1 - epsilon #Maximum possible cos(Theta) value
    
    chi = rand.rand(Num_Events) #uniform random number between 0 and 1

    cos_Thetas = 1 - (1-cos_theta_max)**(1-chi) * (1-cos_theta_min)**chi
    return(cos_Thetas)

#Weighted differential for the integration to account
#   for preferential sampling
def cos_Theta_differential(cos_Theta, epsilon, num_Events):
    '''
    Find the weighted differential for the scattering angle when performing the integral
    args:
        cos_Theta: cosine of the scattering angle (float or array of floats)
        epsilon: 1 - maximum possible value of cos(Theta) (float)
        num_Events: number of events for which we are performing the integral (int)
        
    returns:
        d_cos_Theta: Weighted scattering angle differential for the events 
                    (float or array of floats, same size as cos_Theta)
    '''
    char_val = (2-epsilon)/ np.log(2/epsilon)  #Characteristic value of 1 - cos(Theta)
    d_cos_Theta = (2/num_Events**(1/3)) * (1-cos_Theta)/char_val #Differential for the scattering angle
    
    return(d_cos_Theta)

##FUNCTIONS FOR ARBITRARY SCATTERING DEPENDENCIES

#Function to sample scattering angles from an
#   arbitrary scattering distribution
def Sample_Arbitrary_Scat_Angles(num_events,cos_vals,frac_diff_cross_sec_vals):
    '''
    This function samples scattering angles to best fit a
    distribution of scattering cross-sections
    
    args:
        num_events: number of samples we want (int)
        cos_vals: cosines of the scattering angles at which we have
                data about the cross section (array of floats)
        frac_diff_cross_sec_vals: values of 1/sigma * d_simga/d_cos(Theta)
                at the specified values of cos(Theta) (array of floats, same size as cos_vals)
        
    returns:
        cos_Thetas: array of length num_Events with the sampled scattering
                cross sections.
                
    actions:
        calculates the cdf of the cross section at each angle, selects a
        random value between 0 and 1, finds the angle which has the 
        corresponding cdf
    '''
    #Create a new array for the cosines with -1 and 1 added
    cos_full = np.zeros(len(cos_vals)+2)
    cos_full[0],cos_full[-1] = -1,1
    cos_full[1:-1] = cos_vals
    
    #Create a new array for differential cross sections, extending the current edges to -1 and 1
    cross_sec_full = np.zeros(len(frac_diff_cross_sec_vals)+2)
    cross_sec_full[0], cross_sec_full[-1] = frac_diff_cross_sec_vals[0], frac_diff_cross_sec_vals[-1]
    cross_sec_full[1:-1] = frac_diff_cross_sec_vals
    
    #Create an array for the cdfs
    cdfs = np.zeros(len(cos_full))
    for i in range(1,len(cdfs)):
        cdfs[i] = np.trapz(cross_sec_full[:i+1],cos_full[:i+1])
    '''
    fig1 = plt.figure()
    plt.plot(cos_full, cdfs)
    '''
    #Uniformly sample the cdf
    Ran = rand.rand(num_events)
    #Interpolate to find the corresponding cosine value
    cos_Thetas = np.interp(Ran,cdfs,cos_full)
    return(cos_Thetas)

#Weighted differential for an arbitrary scattering
#   distribution
def Arbitrary_Scattering_Differential(cos_Theta,num_events,cos_vals,frac_diff_cross_sec_vals):
    '''
    This function determines the proper differential for each event
    to perform the integral.
    args:
        cos_Theta: scattering angle of the event(s) (float or array of floats)
        num_events: number of samples we want (int)
        cos_vals: cosines of the scattering angles at which we have
                data about the cross section (array of floats)
        frac_diff_cross_sec_vals: values of 1/sigma * d_simga/d_cos(Theta)
                at the specified values of cos(Theta) (array of floats, same size as cos_vals)
                
    returns:
        d_cos_Theta: properly weighted differential (float of same size as cos_Theta)
    '''
    rho = np.interp(cos_Theta,cos_vals,frac_diff_cross_sec_vals) #Weight at scattering angle
    d_cos_Theta = 1/ (rho * num_events**(1/3))
    
    return(d_cos_Theta)

if __name__ == "__main__":
    main()
