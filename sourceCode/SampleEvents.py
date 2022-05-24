def main():
    print("Hello World")
    return(0)
'''
This module is sampling the properties of our events at the beginning of the 
    Monte Carlo simulation
    
    Functions:
        Sample_Neutrino_Energies: Sample atmospheric neutrino energies according
            to a power law.
        weight_Energy: Calculate the proper weight for integrating over neutrino
            energies when the energies were sampled according to a power law
        Sample_cos_Theta: Sample HNL scattering angle according to a power law of
            (1 - cos(Theta))
        weight_cos_Theta: Calculate the proper weight for integrating over cosine
            of HNL scattering angles when the angles were sampled according to (1 - cos(Theta))
        Sample_Arbitrary_Scat_Angles: Sample HNL scattering angle according to an
            arbitrary distribution (an array of the differential cross section is required)
        Arbitrary_Scattering_Weight: Calculate the weight for integrating over HNL scattering
            angles for an arbitrary distribution.
        Sample_Interaction_Location: Sample an interaction location for the neutrino
            uniformly in space up to a specified maximum distance from the detector
        weight_positions: Calculate weight for integrating over interaction positions
            when positions are sampled uniformly in space
        Sample_Neutrino_Entry_Position: Use the neutrino interaction position and 
            HNL scattering angle to sample the entry position of the neutrino
        Sample_Interaction_Location_Exp: Sample neutrino interaction position according
            to a probability exponentially decaying with distance from the detector
        Weight_Positions_Exp: Calculate weight for integrating over positions when sampled
            as an exponential decay in the distance from the detector
        energyOut: Calculate the energy of the HNL after the scattering event
'''

#Initialization
import numpy as np 
from numpy import random as rand
from numpy import sin, cos
from numpy import pi as pi
import matplotlib 
from matplotlib import pyplot as plt
from scipy import special as spc

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



def Sample_cos_Theta(Num_Events, epsilon, Theta_max = pi, power = 1 ):
    '''
    Samples cosines of the scattering angles according to a 1/(1-cos(Theta)) distribution
    
    args:
        Num_Events: number of scattering angles that we wish to sample (int)
        epsilon: our value of 1 - cos(Theta_min)
        Theta_max: maximum scattering angle possible for the interaction (float)
        power: We wish to sample the scattering angles for a 
            distribution (1 - cos(Theta))^(-1*power)
    
    returns:
        cos_Thetas: sampled scattering angles (array of length Num_Events, floats)
    '''
    cos_Theta_min = 1 - epsilon 
    cos_Theta_max = cos(Theta_max)
    if power == 1:
        
        
        chi = rand.rand(Num_Events)
    
        cos_Thetas = 1 - (1-cos_Theta_min)**(1-chi) * (1-cos_Theta_max)**chi

        
    else:
        norm_const = (1-power) / ((1- cos_Theta_min)**(1-power) - (1 - cos_Theta_max)**(1-power))
        
        chi = rand.rand(Num_Events)
        
        one_minus_cos_Theta = ((1-power)/norm_const * chi + (1 - cos_Theta_max))**(1/(1-power))
        
        cos_Thetas = 1 - one_minus_cos_Theta
        
    return(cos_Thetas)

def weight_cos_Theta(cos_Theta, epsilon, Theta_max = pi, power = 1):
    '''
    Determines the correct weights for our preferential sampling of scattering angles
    
    args:
        cos_Theta: cosine of the scattering angle for a particular event (float or array of floats)
        epsilon: our value of 1 - cos(Theta_min)
        Theta_max: maximum scattering angle possible for the interaction (float)
        power: We wish to sample the scattering angles for a 
            distribution (1 - cos(Theta))^(-1*power)
    
    returns:
        w_Theta: the proper weight for the event (float same size as cos_Theta)
    '''
    cos_Theta_min = 1 - epsilon 
    cos_Theta_max = cos(Theta_max)
    if power == 1:
        
        
        w_Theta = np.log((1-cos_Theta_max)/(1-cos_Theta_min)) * (1-cos_Theta)/(cos_Theta_min - cos_Theta_max)
        
    else:
        norm_const = (1-power) / ((1- cos_Theta_min)**(1-power) - (1 - cos_Theta_max)**(1-power))
        w_Theta = (1/norm_const) * (1 - cos_Theta)**power / (cos_Theta_max - cos_Theta_min)
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

def Sample_Interaction_Locations(Num_Events, Y, R_max, R_min):
    '''
    Sample locations within the Earth for neutrino dipole interactions.  Sample uniformly in
        volume up to a distance R_max from the detector.
    
    args:
        Num_Events: Number of events that we would like to sample (int)
        Y: Cartesian coordinates of the detector location in units of R_Earth (3 element float array)
        R_max: Maximum distance for which we care about dipole interactions in units of R_Earth
            (float array with Num_Events elements)
        R_min: Minimum distance for which we care about dipole interaction
            in units of R_Earth = 1 (float array with Num_Events elements)
    
    returns:
        x_vect_vals: Num_Events-by-3 array of the sampled locations for the neutrino
                dipole interactions in units of R_Earth
    '''
    
    R_Earth = 1   # 6378.1 * 1000* 100    #Radius of the Earth (cm)
    
    x_vect_vals = np.zeros((Num_Events,3)) #positions of interactions (cm)
    needed_indeces = x_vect_vals[:,0] < 1 #indeces for which we still need to assign positions
    needed_events = Num_Events
    x_mags = np.zeros(Num_Events)
    
    while needed_events > 0:
        new_Rmaxs = np.minimum(2*R_Earth,R_max[needed_indeces])
        r_primes = ((rand.rand(needed_events) * (new_Rmaxs**3 - R_min**3)) + R_min**3)**(1/3)#cm
        cos_theta_primes = 1 - 2 * rand.rand(needed_events)
        theta_primes = np.arccos(cos_theta_primes)
        phi_primes = 2*pi*rand.rand(needed_events)

        #Find the vector from the spherical coordinate vals
        x_vect_vals[needed_indeces,0] = Y[0] + r_primes*sin(theta_primes)*cos(phi_primes)
        x_vect_vals[needed_indeces,1] = Y[1] + r_primes*sin(theta_primes)*sin(phi_primes)
        x_vect_vals[needed_indeces,2] = Y[2] + r_primes*cos(theta_primes)
        
        x_mags = np.sqrt(x_vect_vals[:,0]**2 + x_vect_vals[:,1]**2 + x_vect_vals[:,2]**2)
        needed_indeces = (x_mags > R_Earth) 
        needed_events = sum(needed_indeces)
        #print(100 - 100 * needed_events/Num_Events, "% done")
        
    return(x_vect_vals)

def weight_positions(Y,R_max, R_min):
    '''
    Computes the proper weight for position sampling. Sample is uniform in
        volume up to a distance R_max from the detector.
    args:
        Y: Cartesian coordinates of the detector location in units of R_Earth (3 element float array)
        R_max: Maximum distance for which we care about dipole interactions in units of R_Earth
            (float array with Num_Events elements)
    returns:
        w_V: Positional weight
    '''
    
    R_Earth = 1 #6378.1 * 1000* 100    #Radius of the Earth (cm)
    V_Earth = 4*pi/3 * R_Earth**3
    V_int = 0
    Y_mag = np.sqrt(np.dot(Y,Y))
    
    
    V_int += 4*pi/3 * R_max**3 * np.heaviside(R_Earth - R_max - Y_mag,1)
    
    V_int += 4*pi/3 * R_Earth**3 * np.heaviside(R_max - R_Earth - Y_mag,0)
    
    V_int += pi/(12*Y_mag) * ((R_Earth + R_max - Y_mag)**2 
                      * (Y_mag**2 + 2*Y_mag*(R_Earth+R_max) - 3*(R_Earth - R_max)**2)
                      *np.heaviside(R_max + Y_mag - R_Earth,0) * np.heaviside(R_Earth + Y_mag - R_max,1))
    
    V_int -= 4*pi/3 * R_min**3
    w_V = V_int / V_Earth
    return(w_V)

def Sample_Neutrino_Entry_Position(X, Y, cos_Theta):
    
    '''
    Samples the location where the neutrino entered the Earth
    
    args:
        X: Cartesian coordinates of the neutrino interaction in units of R_Earth
            (number of events-by-3 array of floats)
        Y: Cartesian coordinates of the detector position in units of R_Earth
            (3 element array of floats)
        cos_Theta: scattering angles for the neutrino interactions
            (array of floats of length number of events)
    
    returns:
        W: Cartesian coordinates of the neutrino entry position in units of R_Earth
            (number of events-by-3 array of floats.)
    '''
    v_1_hat = np.zeros((len(cos_Theta),3))
    v_2_hat = np.zeros((len(cos_Theta),3))
    v_in_hat = np.zeros((len(cos_Theta),3))
    psi = 2*pi*rand.rand(len(cos_Theta))
    Theta = np.arccos(cos_Theta)
    R_Earth = 1    #6378.1 * 1000* 100    #Radius of the Earth (cm)
    
    X_mag = np.sqrt(X[:,0]**2 + X[:,1]**2 + X[:,2]**2)
    
    X_minus_Y = X-Y
    X_minus_Y_mag = np.sqrt(X_minus_Y[:,0]**2 + X_minus_Y[:,1]**2 + X_minus_Y[:,2]**2)
    
    X_minus_Y_hat = np.zeros((len(cos_Theta),3))
    X_minus_Y_hat[:,0] = X_minus_Y[:,0]/X_minus_Y_mag
    X_minus_Y_hat[:,1] = X_minus_Y[:,1]/X_minus_Y_mag
    X_minus_Y_hat[:,2] = X_minus_Y[:,2]/X_minus_Y_mag
    
    X_cross_Y = np.cross(X,Y)
    X_cross_Y_mag = np.sqrt(X_cross_Y[:,0]**2 + X_cross_Y[:,1]**2 + X_cross_Y[:,2]**2)
    
    
    v_1_hat[:,0] = X_cross_Y[:,0]/X_cross_Y_mag
    v_1_hat[:,1] = X_cross_Y[:,1]/X_cross_Y_mag
    v_1_hat[:,2] = X_cross_Y[:,2]/X_cross_Y_mag
    
    
    v_2_hat = np.cross(X_minus_Y_hat, v_1_hat)
    
    v_in_hat[:,0] = (-X_minus_Y[:,0]/X_minus_Y_mag * cos(Theta) + v_1_hat[:,0] * sin(Theta) * cos(psi)
                     + v_2_hat[:,0] * sin(Theta) * sin(psi) )
    v_in_hat[:,1] = (-X_minus_Y[:,1]/X_minus_Y_mag * cos(Theta) + v_1_hat[:,1] * sin(Theta) * cos(psi)
                     + v_2_hat[:,1] * sin(Theta) * sin(psi) )
    v_in_hat[:,2] = (-X_minus_Y[:,2]/X_minus_Y_mag * cos(Theta) + v_1_hat[:,2] * sin(Theta) * cos(psi)
                     + v_2_hat[:,2] * sin(Theta) * sin(psi) )
    
    
    
    X_dot_v_in_hat = X[:,0] * v_in_hat[:,0] + X[:,1] * v_in_hat[:,1] + X[:,2] * v_in_hat[:,2]
    
    v_in_mag = X_dot_v_in_hat + np.sqrt(X_dot_v_in_hat**2 +R_Earth**2 - X_mag**2)
    
    W = np.zeros((len(cos_Theta),3))
    W[:,0] = X[:,0] - v_in_hat[:,0] * v_in_mag
    W[:,1] = X[:,1] - v_in_hat[:,1] * v_in_mag
    W[:,2] = X[:,2] - v_in_hat[:,2] * v_in_mag
    return(W)

def Sample_Interaction_Location_Exp(Num_Events, Y, R_min, N_lambdas):
    '''
    Sample locations within the Earth for neutrino dipole interactions taking exponential
        decay into account
    
    args:
        Num_Events: Number of events that we would like to sample (int)
        Y: Cartesian coordinates of the detector location in units of R_Earth (3 element float array)
        R_min: Minimum distance for which we care about dipole interaction
            in units of R_Earth = 1 (float array with Num_Events elements)
        N_lambdas: Decay lengths of the HNL in units of R_Earth (array of floats)
    
    returns:
        x_vect_vals: Num_Events-by-3 array of the sampled locations for the neutrino
                dipole interactions in units of R_Earth
    '''
    R_Earth = 1   # 6378.1 * 1000* 100    #Radius of the Earth (cm)
    Y_mag = np.sqrt(np.dot(Y,Y))
    R_max = R_Earth + Y_mag
    
    x_vect_vals = np.zeros((Num_Events,3)) #positions of interactions (cm)
    needed_indeces = x_vect_vals[:,0] < 1 #indeces for which we still need to assign positions
    needed_events = Num_Events
    x_mags = np.zeros(Num_Events)
    

    all_r_primes = R_min - N_lambdas * np.log(1 - rand.rand(Num_Events) *(1 - np.exp((R_min-R_max)/N_lambdas)))
    
    
    while needed_events > 0:
        
        r_primes = all_r_primes[needed_indeces]

        
        #max_cosines = 1 * np.heaviside(R_Earth - r_primes,1) + -r_primes/(2*R_Earth) * np.heaviside(r_primes-R_Earth,0)
        max_cosines = 1 * np.heaviside(R_Earth - Y_mag - r_primes,1)
        max_cosines += (((R_Earth**2 - Y_mag**2 - r_primes**2)/(2* Y_mag*r_primes)) 
                        * np.heaviside(Y_mag + r_primes - R_Earth,0))
        cos_theta_primes = -1 + (max_cosines+1)* rand.rand(needed_events)

        theta_primes = np.arccos(cos_theta_primes)
        phi_primes = 2*pi*rand.rand(needed_events)

        #Find the vector from the spherical coordinate vals
        x_vect_vals[needed_indeces,0] = Y[0] + r_primes*sin(theta_primes)*cos(phi_primes)
        x_vect_vals[needed_indeces,1] = Y[1] + r_primes*sin(theta_primes)*sin(phi_primes)
        x_vect_vals[needed_indeces,2] = Y[2] + r_primes*cos(theta_primes)
        
        x_mags = np.sqrt(x_vect_vals[:,0]**2 + x_vect_vals[:,1]**2 + x_vect_vals[:,2]**2)
        
        
        det_dist = np.sqrt((Y[0]-x_vect_vals[:,0])**2 + (Y[1]-x_vect_vals[:,1])**2 
                           + (Y[2] - x_vect_vals[:,2])**2)
        needed_indeces = np.any([x_mags > R_Earth, det_dist < R_min],axis = 0) 
        
        needed_events = sum(needed_indeces)
        
    return(x_vect_vals)

def Weight_Positions_Exp(Y,R_min,x_vect_vals,N_lambdas):
    '''
    Weights interaction positions for integration taking exponential decay into account
    
    args:
        Y: Cartesian coordinates of the detector location in units of R_Earth (3 element array)
        R_min: Minimum distance for which we care about dipole interaction in units of
            R_Earth = 1 (float)
        x_vect_vals: n-by-3 array of the sampled locations for the neutrino dipole
            interactions in units of R_Earth
        N_lambdas: Decay lengths of the HNL in units of R_Earth (array of floats)
        
    returns:
        w_V: Positional weight (array of floats)
    '''
    
    R_Earth = 1
    Y_mag = np.sqrt(np.dot(Y,Y))
    R_max = R_Earth + Y_mag
    
    V_Earth = 4*pi/3 * R_Earth**3
    V_min = 4*pi/3 * R_min**3
    r_primes_mag = np.sqrt((Y[0] - x_vect_vals[:,0])**2 + (Y[1] - x_vect_vals[:,1])**2
                          + (Y[2] - x_vect_vals[:,2])**2)
    
    dV_dr = 0
    dV_dr += 4*pi*r_primes_mag**2 * np.heaviside(R_Earth - Y_mag - r_primes_mag, 1)
    
    dV_dr += ((-pi * r_primes_mag)/Y_mag * (r_primes_mag - R_Earth - Y_mag) 
              * (r_primes_mag + R_Earth - Y_mag)) * np.heaviside(Y_mag + r_primes_mag - R_Earth,0)
    
    total_R = Y_mag + R_Earth
    
    w_V = (N_lambdas *(1 - np.exp((R_min - R_max)/N_lambdas)) 
           * np.exp((r_primes_mag - R_min)/N_lambdas) * 1/(R_max - R_min) * dV_dr * total_R)
    
    w_V = w_V / (4/3 * pi)
    
    return(w_V)

def energyOut(Enu,cos_Theta,mN,M_target,branch=1):
    '''
    args: 
        Enu: Energy of incident neutrino
        cos_Theta: scattering angle  
        mN       : mass of upscattered particle
        M_target : mass of target particle (assumed stationary in the lab frame)
        branch   : branch of multi-valued solution desired 
                                                   default =1 (high energy) 
                                                   alternative option = 2 (low energy) 
    returns: 
        EN : energy of HNL after scattering event
    '''

    M=M_target
    if mN>0:
        kappa= (mN - 4*Enu*M*mN**2 - 4*M**2*mN**2 + 4*Enu**2*(M**2 - mN**2))/(4.*Enu**2*mN**2)
    elif mN==0:
        kappa=1
    else:
        assert mN<=0,"HNL mass cannot be negative"

    # parameters of relevant quadratic equation for E_nu 
    a=4*(-((-1 + cos_Theta**2)*Enu**2) + 2*Enu*M + M**2)
    b=-4*(Enu + M)*(2*Enu*M + mN**2)
    c=4*cos_Theta**2*Enu**2*mN**2 + (2*Enu*M + mN**2)**2
    
    if kappa>=0:
        if cos_Theta>0:
            return( (-b+np.sqrt(b**2-4*a*c))/(2*a) )
        elif cos_Theta<=0:
            return( (-b-np.sqrt(b**2-4*a*c))/(2*a) )
    if kappa<0:
        if branch==1:
            return( (-b+np.sqrt(b**2-4*a*c))/(2*a) )
        elif branch==2:
            return( (-b-np.sqrt(b**2-4*a*c))/(2*a) )
        else:
            print("You did not input a branch value of 1 or 2")
            return(0)


if __name__ == "__main__":
    main()
