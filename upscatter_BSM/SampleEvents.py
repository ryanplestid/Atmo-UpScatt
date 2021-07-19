def main():
    print("Hello World")
    return(0)

#Initialization
import numpy as np #Package for array functions
import formFactorFit
from numpy import random as rand
from numpy import sin, cos
from numpy import pi as pi

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

def weight_Energy(En, E_min, E_max, power_law):
    '''
    This function calculates the proper weights to perform the integral
    over neutrino energies
    
    args:
        En: Energy of the neutrino in GeV (float or array of floats)
        E_min: minimum Energy of our distribution in GeV (float)
        E_max: maximum Energy of our distriubtion in GeV (float)
        power_law: our distibution follows E^{-power_law} (float)
    
    returns:
        w_Energy: proper weight for event (float same size as En)
    '''
    kappa = (1 - power_law)/ (E_max**(1-power_law) - E_min**(1-power_law))
    w_Energy = En**power_law / (kappa * (E_max - E_min))
    return(w_Energy)

def Sample_cos_Theta(Num_Events, epsilon, Theta_max = pi ):
    '''
    Samples cosines of the scattering angles according to a 1/(1-cos(Theta)) distribution
    
    args:
        Num_Events: number of scattering angles that we wish to sample (int)
        epsilon: our value of 1 - cos(Theta_min)
        Theta_max: maximum scattering angle possible for the interaction (float)
    
    returns:
        cos_Thetas: sampled scattering angles (array of length Num_Events, floats)
    '''
    cos_Theta_min = 1 - epsilon 
    cos_Theta_max = cos(Theta_max)
    
    chi = rand.rand(Num_Events)

    cos_Thetas = 1 - (1-cos_Theta_min)**(1-chi) * (1-cos_Theta_max)**chi
    return(cos_Thetas)

def weight_cos_Theta(cos_Theta, epsilon, Theta_max = pi):
    '''
    Determines the correct weights for our preferential sampling of scattering angles
    
    args:
        cos_Theta: cosine of the scattering angle for a particular event (float or array of floats)
        epsilon: our value of 1 - cos(Theta_min)
        Theta_max: maximum scattering angle possible for the interaction (float)
    
    returns:
        w_Theta: the proper weight for the event (float same size as cos_Theta)
    '''
    cos_Theta_min = 1 - epsilon 
    cos_Theta_max = cos(Theta_max)
    
    w_Theta = np.log((1-cos_Theta_max)/(1-cos_Theta_min)) * (1-cos_Theta)/(cos_Theta_min - cos_Theta_max)
    
    return(w_Theta)

#Arbitrary Scattering Functions
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

def Arbitrary_Scattering_Weight(cos_Theta,cos_vals,frac_diff_cross_sec_vals):
    '''
    This function determines the proper differential for each event
    to perform the integral.
    args:
        cos_Theta: scattering angle of the event(s) (float or array of floats)
        cos_vals: cosines of the scattering angles at which we have
                data about the cross section (array of floats)
        frac_diff_cross_sec_vals: values of 1/sigma * d_simga/d_cos(Theta)
                at the specified values of cos(Theta) (array of floats, same size as cos_vals)
                
    returns:
        w_Theta: proper weight for event (float of same size as cos_Theta)
    '''
    rho = np.interp(cos_Theta,cos_vals,frac_diff_cross_sec_vals)
    w_Theta = 2/ (rho)
    
    return(w_Theta)

if __name__ == "__main__":
    main()