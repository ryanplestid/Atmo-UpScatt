def main():
    print('Hello World')
    return(0)

#Initialization
import numpy as np #Package for array functions
from numpy import random as rand
from numpy import sin, cos
from numpy import pi

def Calc_cos_zeta_prime(num_events,alpha_decay):
    '''
    Returns the cosine of the angle between the lepton and
    decay photon in the rest frame.
    
    args:
        num_events: number of angles that we wish to sample (int)
        alpha_decay: value of alpha determining if the lepton is a
                    Dirac or Majoranna particle [-1,1] (float)
    
    returns:
        cos_zeta_primes: array of the cosines of the angles between
                    the lepton direction of travel and photon direction
                    in the rest frame
    '''
    #Sample uniformly if alpha = 0
    if alpha_decay == 0:
        cos_zeta_primes = 2*rand.rand(num_events) -1
        return(cos_zeta_primes)
    
    #Sample according to the correct distribution if alpha !=0
    chi = rand.rand(num_events)
    cos_zeta_primes = (1/alpha_decay)* (-1+ np.sqrt(1-alpha_decay*(2-alpha_decay-4*chi)))
    
    return(cos_zeta_primes)

def Calc_Zetas_and_Energies(cos_zeta_prime,En,mn):
    '''
    Calculates the angle and energy of the photons in the
    lab frame
    
    args:
        cos_zeta_prime: cosine of the angle between photon and lepton 
                    directions in the rest frame (float or array of floats)
        En: Energies of leptons in GeV (float, same size as cos_zeta_prime)
        mn: Mass of the heavy neutral lepton in GeV (float)
    
    returns:
        zeta: angle between the lepton and photon directions in
            the lab frame (float, same size as En)
        E_gamma: Energy of the photon in the lab frame in GeV
                (float, same size as En)
    '''
    #Calculate the sine of zeta prime
    sin_zeta_prime = np.sqrt(1-cos_zeta_prime**2)
    
    #Calculate the tangent of zeta
    tan_zeta = (mn/En) * sin_zeta_prime/ (cos_zeta_prime + np.sqrt(1 - mn**2/En**2))
    
    #Get zetas from tan(zeta), make sure it's in the right quadrant
    zeta = (np.arctan(tan_zeta) + pi*np.heaviside(-tan_zeta,0) 
            + pi * np.heaviside(-cos_zeta_prime - 1,1))
    
    #Calculate the Energy of the photon in the lab frame
    E_gamma = (En/2) * (1+np.sqrt(1 - mn**2/En**2)*cos_zeta_prime)
    
    return(zeta, E_gamma)

def Calc_cos_phi_det(Y, X, zeta):
    '''
    Calculates the cosine of the angle between the photon direction and the zenith of the detector
    
    args:
        Y: 3 element array of the cartesian coordinates of the detector in cm
        X: n-by-3 array of the Cartesian coordinates of the n neutrino
            interaction positions in cm
        zeta: array of n floats for the scattering angle of the photons in the lab frame
        
    returns:
        cos_phi_det: cosine of the angle between the photon direction and zenith angle of the detector
    '''
    psis = 2*pi*rand.rand(len(zeta))
    
    Y_mag = np.sqrt(np.dot(Y, Y))
    
    
    X_mag = np.sqrt(X[:,0]**2 + X[:,1]**2 + X[:,2]**2)
    
    Y_minus_X_mag = np.sqrt((Y[0] - X[:,0])**2 + (Y[1] - X[:,1])**2 + (Y[2] - X[:,2])**2)
    
    X_cross_Y = np.cross(X,Y)
    
    X_cross_Y_mag = np.sqrt(X_cross_Y[:,0]**2 + X_cross_Y[:,1]**2 + X_cross_Y[:,2]**2)
    
    Y_dot_X = Y[0]*X[:,0] + Y[1]*X[:,1] + Y[2]*X[:,2]
    '''
    except:
        X_mag = np.sqrt(X[0]**2 + X[1]**2 + X[2]**2)
        
        Y_minus_X_mag = np.sqrt((Y[0] - X[0])**2 + (Y[1] - X[1])**2 + (Y[2] - X[2])**2)
        
        X_cross_Y = np.cross(X,Y)
        
        X_cross_Y_mag = np.sqrt(X_cross_Y[0]**2 + X_cross_Y[1]**2 + X_cross_Y[2]**2)
        
        Y_dot_X = Y[0]*X[0] + Y[1]*X[1] + Y[2]*X[2]
    '''
    
    first_term = (Y_mag**2 - Y_dot_X)/(Y_mag*Y_minus_X_mag)
    second_term = (X_mag**2 * Y_mag**2 - Y_dot_X**2)/(Y_mag * X_cross_Y_mag * Y_minus_X_mag)

    
    cos_phi_det = first_term* cos(zeta) + second_term*sin(zeta)*sin(psis)
    
    return(cos_phi_det)

def Rate_In_Each_Bin(min_E, max_E, num_E_bins, num_cos_bins, E_gamma, cos_phi_det, dR):
    '''
    Calculates the rate of photons observed in each detector, binned by
        the energies and angles relative to the detector zenith
    
    args:
        min_E: minimum photon energy considered in GeV (float)
        max_E: maximum photon energy considered in GeV (float)
        num_E_bins: number of bins in energy (int)
        num_cos_bins: number of bins in angle (int)
        E_gamma: Energies of the photons in GeV (float or array of floats)
        cos_phi_det: Cosine of the angle between the photon direction and
                the zenith of the detector (float, same size as E_gamma)
        dR: rate that the specific event contributes (float, same size as E_gamma)
        
    returns:
        E_midpoints: Midpoints of the energy bins
        cos_midpoints: midpoints of the cosines of the angular bins
        rates: 2D array of the rates in each bin in s^-1.  First index corresponds 
            to the energy bin, second index corresponds to the angular bin (float)
    '''
    
    
    E_edges = np.linspace(min_E, max_E, num_E_bins + 1)
    cos_edges = np.linspace(-1,1, num_cos_bins + 1)
    
    E_midpoints = (0.5) * (E_edges[0:-1] + E_edges[1:])
    cos_midpoints = 0.5 * (cos_edges[0:-1] + cos_edges[1:])
    
    rates = np.zeros((len(E_midpoints), len(cos_midpoints)))
    
    for E_index in range(len(E_midpoints)):
        upper_E = E_edges[E_index + 1]
        lower_E = E_edges[E_index]
        for cos_index in range(len(cos_midpoints)):
            upper_cos = cos_edges[cos_index + 1]
            lower_cos = cos_edges[cos_index]
            
            rates[E_index,cos_index] = sum(dR *(np.heaviside(E_gamma-lower_E,1) * np.heaviside(upper_E - E_gamma,0)
                                            *np.heaviside(cos_phi_det-lower_cos,1) 
                                            * np.heaviside(upper_cos - cos_phi_det,0)))
    
    return(E_midpoints,cos_midpoints,rates)

if __name__ == '__main__':
    main()
