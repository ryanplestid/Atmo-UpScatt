import unittest
import numpy as np
from numpy import random as rand
import matplotlib
from matplotlib import pyplot as plt

import dipoleModel
import atmoNuIntensity

class Testing_Dipole_Functions(unittest.TestCase):
    
    #Test that the cross section is always non-negative
    def test_cross_sec_pos(self):
        for k in range(1000):
            #Select random parameters
            d = 1e-9 * rand.rand() #Dipole coupling MeV^-1
            mn = rand.rand() #Lepton Mass GeV
            En = 100*rand.rand() #Neutrino Energy GeV
            cos_Theta = 2*rand.rand() - 1 #Cosine Scattering angle
            Zeds = (30*rand.rand(3) + 1).astype(int) #Nuclear atomic numbers
            R1s = 5*rand.rand(3) #Helm effective radius (fm)
            Ss = 1*rand.rand(3) #Helm skin thickness (fm)
            fracs = rand.rand(3)
            fracs = fracs/sum(fracs) #Fractional number desity of each nucleus
            
            val = dipoleModel.Full_d_sigma_d_cos_Theta(d,mn,En,cos_Theta,Zeds,R1s,Ss,fracs)
            
            self.assertGreaterEqual(val,0)
    
    #Test that the cross section vanishes if m_N > E_N
    def test_cross_sec_vanish(self):
        for k in range(1000):
            #Select random parameters
            d = 1e-9 * rand.rand() #Dipole coupling MeV^-1
            mn = rand.rand() #Lepton Mass GeV
            En = rand.rand() #Neutrino Energy GeV
            cos_Theta = 2*rand.rand() - 1 #Cosine Scattering angle
            Zeds = (30*rand.rand(3) + 1).astype(int) #Nuclear atomic numbers
            R1s = 5*rand.rand(3) #Helm effective radius (fm)
            Ss = 1*rand.rand(3) #Helm skin thickness (fm)
            fracs = rand.rand(3)
            fracs = fracs/sum(fracs) #Fractional number desity of each nucleus
            
            if mn > En:
                val = dipoleModel.Full_d_sigma_d_cos_Theta(d,mn,En,cos_Theta,Zeds,R1s,Ss,fracs)
                self.assertEqual(val,0)
    
    #Test that the decay length is 0 when m_N > E_N
    def test_decay_length(self):
        for k in range(1000):
            d = 1e-9*rand.rand() #Dipole coupling MeV^-1
            En = rand.rand() #Neutrino Energies GeV
        
            #Make lepton masses greater than the energy
            mn = En + rand.rand() #lepton masses GeV
            val = dipoleModel.decay_length(d,mn,En)
            self.assertEqual(val,0)
    
    #Test that our sampled scattering angles follow a
    # 1/(1-cos) distribution
    def test_sampling_angles(self):
        #Sample Scattering angle
        epsilon = 1e-6
        cos_Thetas = dipoleModel.Sample_cos_Theta(int(1e7),epsilon)
        
        #Make histogram of sampled events
        n,bins,patches = plt.hist(1-cos_Thetas,400)
        midpoints = (bins[1:]+bins[:-1])/2
        
        #Fit a line to the log-log plot of the histogram
        slope,intercept = np.polyfit(np.log(midpoints),np.log(n),1)
        
        self.assertAlmostEqual(slope,-1,delta = 0.1)

class Testing_Flux_Functions(unittest.TestCase):
    
    #Confirm that the sampling follows the power law
    def test_sampling_energies(self):
        num_Events = int(1e7)
        E_min = 0.1 #Minimum Neutrino Energy GeV
        #The max neutrino energy used here is lower than the
        #   true max.  We want to avoid 0s in the histogram
        E_max = 20 #Maximum Neutrino Energy GeV
        power_law = 1 + 2*rand.rand()
        #Make sure that power_law isn't equal to 1
        while power_law == 1:
            power_law = 1 + 2*rand.rand()
        
        #Sample Energies
        En = atmoNuIntensity.Sample_Neutrino_Energies(num_Events,E_min,E_max,power_law)
        
        n,bins,patches = plt.hist(En,50)
        midpoints = (bins[1:]+bins[:-1])/2
        
        #Fit a line to the log-log plot of the histogram
        slope,intercept = np.polyfit(np.log(midpoints),np.log(n),1)
        
        #Check that the plot has the correct dependence
        self.assertAlmostEqual(slope, -1*power_law,delta = 0.1)
    
    #Confirm that the sampling and weighted differential
    # return a uniform distibution
    def test_weighting_energies(self):
        num_Events = int(1e7)
        E_min = 0.1 #Minimum Neutrino Energy GeV
        #The max neutrino energy used here is lower than the
        #   true max.  We want to avoid 0s in the histogram
        E_max = 20 #Maximum Neutrino Energy GeV
        power_law = 1 + 2*rand.rand()
        #Make sure that power_law isn't equal to 1
        while power_law == 1:
            power_law = 1 + 2*rand.rand()
        
        #Sample Energies
        En = atmoNuIntensity.Sample_Neutrino_Energies(num_Events,E_min,E_max,power_law)
        
        n,bins,patches = plt.hist(En,50)
        midpoints = (bins[1:]+bins[:-1])/2
        
        #Calculate weighted differential dE at midpoints
        dE = atmoNuIntensity.Calculate_dE(num_Events,midpoints,E_min,E_max,power_law)
        
        #Check that the number of samples times dE gives uniform dist
        slope,intercept = np.polyfit(np.log(midpoints),np.log(n*dE),1)
        self.assertAlmostEqual(slope, 0, delta = 0.1)
        

if __name__ == "__main__":
    unittest.main()