import unittest
import numpy as np
from numpy import random as rand
from numpy import pi
import matplotlib
from matplotlib import pyplot as plt

import dipoleModel
#import atmoNuIntensity
import SampleEvents
import DetectorModule
import Mass_Mixing_Model

class Testing_Dipole_Functions(unittest.TestCase):
    
    #Test that the cross section is always non-negative
    def test_cross_sec_pos(self):
        for k in range(1000):
            #Select random parameters
            d = 1e-9 * rand.rand() #Dipole coupling MeV^-1
            mn = 0.93*rand.rand() #Lepton Mass GeV
            En = 100*rand.rand() #Neutrino Energy GeV
            cos_Theta = 2*rand.rand() - 1 #Cosine Scattering angle
            Zeds = (30*rand.rand(3) + 1).astype(int) #Nuclear atomic numbers
            As = (30*rand.rand(3) + Zeds).astype(int)
            R1s = 5*rand.rand(3) #Helm effective radius (fm)
            Ss = 1*rand.rand(3) #Helm skin thickness (fm)
            num_dens = 1e23 * rand.rand(3)
            
            val = dipoleModel.Full_N_d_sigma_d_cos_Theta(d,mn,En,cos_Theta,Zeds,As,R1s,Ss,num_dens)
            
            self.assertGreaterEqual(val,0)
        print('first done')
    
    #Test that the cross section vanishes if m_N > E_N
    def test_cross_sec_vanish(self):
        for k in range(1000):
            #Select random parameters
            d = 1e-9 * rand.rand() #Dipole coupling MeV^-1
            mn = 0.93*rand.rand() #Lepton Mass GeV
            En = rand.rand() #Neutrino Energy GeV
            cos_Theta = 2*rand.rand() - 1 #Cosine Scattering angle
            Zeds = (30*rand.rand(3) + 1).astype(int) #Nuclear atomic numbers
            As = (30*rand.rand(3) + 1).astype(int)
            R1s = 5*rand.rand(3) #Helm effective radius (fm)
            Ss = 1*rand.rand(3) #Helm skin thickness (fm)
            num_dens = 1e23*rand.rand(3)
            
            if mn > En:
                val = dipoleModel.Full_N_d_sigma_d_cos_Theta(d,mn,En,cos_Theta,Zeds,As,R1s,Ss,num_dens)
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
        print('last done')
        
class Testing_Mass_Mixing_Functions(unittest.TestCase):
    #Show that the cross section always gives a non-negative number
    #   for coherent scattering
    def test_cross_sec_pos_MM_coh(self):
        for trial in range(1000):
            U = rand.rand()
            mn = 0.93*rand.rand()
            E_nu = 100*rand.rand(1)
            cos_Theta = 2*rand.rand(1) -1
            r = rand.rand(1)
            val_coh = Mass_Mixing_Model.N_Cross_Sec_from_radii(U,mn,E_nu,\
                                                               cos_Theta,r,'coherent')

            self.assertGreaterEqual(val_coh,0)
    
    #Check that the cross sec always gives a non-negative number
    #   for incoherent and DIS scattering, when I sample energies and
    #   angles according to the proper distribution (without this sampling,
    #   we have trouble working out scattering dynamics).
    def test_cross_sec_pos_MM_nuc(self):
        for trial in range(1000):
            U = rand.rand()
            m_nuc = 0.94
            m_N = 0.93*rand.rand()
            E_thresh = ((m_nuc * m_N**2) + m_N * (2* m_nuc**2 - m_N**2))/(2 * m_nuc**2 - 2* m_N**2)
            E_nu = 100*rand.rand(1) + E_thresh
            kappa = (4* E_nu**2 * (m_nuc**2 - m_N**2) - 4*E_nu*m_nuc*m_N**2 - 4*m_nuc**2 * m_N**2 + m_N**4) \
                    /(4 * E_nu**2 * m_N**2)
                
            cos_Theta_max = -1 * np.heaviside(kappa,1) + (np.sqrt(abs(kappa))+.001)* np.heaviside(-kappa,0)
            
            cos_Theta = (1 - cos_Theta_max) * rand.rand(1) + cos_Theta_max
            
            r = rand.rand(1)
            val_nuc = Mass_Mixing_Model.N_Cross_Sec_from_radii(U,m_N,E_nu,\
                                                               cos_Theta,r,'nucleon')
            val_DIS = Mass_Mixing_Model.N_Cross_Sec_from_radii(U,m_N,E_nu,\
                                                               cos_Theta,r,'DIS')
            
            self.assertGreaterEqual(val_nuc,0)
            self.assertGreaterEqual(val_DIS,0)

    
    #Show that the cross section is zero if m_N > E_nu
    def test_cross_sec_vanish_MM(self):
        for trial in range(1000):
            U = rand.rand()
            E_nu = 100*rand.rand(1)
            mn = E_nu + rand.rand()
            cos_Theta = 2*rand.rand(1) -1
            r = rand.rand(1)
            val_coh = Mass_Mixing_Model.N_Cross_Sec_from_radii(U,mn,E_nu,\
                                                               cos_Theta,r,'coherent')
            val_nuc = Mass_Mixing_Model.N_Cross_Sec_from_radii(U,mn,E_nu,\
                                                               cos_Theta,r,'nucleon')
            val_dis = Mass_Mixing_Model.N_Cross_Sec_from_radii(U,mn,E_nu,\
                                                               cos_Theta,r,'DIS')
            self.assertEqual(val_coh,0)
            self.assertEqual(val_nuc,0)
            self.assertEqual(val_dis,0)


class Testing_Sampling_Functions(unittest.TestCase):
    
    
    #Test that our sampled scattering angles follow a
    # 1/(1-cos) distribution
    def test_sampling_angles(self):
        #Sample Scattering angle
        epsilon = 1e-6
        cos_Thetas = SampleEvents.Sample_cos_Theta(int(1e7),epsilon)
        
        #Make histogram of sampled events
        n,bins,patches = plt.hist(1-cos_Thetas,400)
        midpoints = (bins[1:]+bins[:-1])/2
        
        #Fit a line to the log-log plot of the histogram
        slope,intercept = np.polyfit(np.log(midpoints),np.log(n),1)
        print('angles done')
        
        self.assertAlmostEqual(slope,-1,delta = 0.1)

    
    #Confirm that the sampling follows the power law
    def test_sampling_energies(self):
        num_Events = int(1e7)
        E_min = 0.1 #Minimum Neutrino Energy GeV
        #The max neutrino energy used here is lower than the
        #   true max.  We want to avoid 0s in the histogram
        E_max = 20 #Maximum Neutrino Energy GeV
        power_law = 1 + 1.5*rand.rand()
        #Make sure that power_law isn't equal to 1
        while power_law == 1:
            power_law = 1 + 1.5*rand.rand()
        
        #Sample Energies
        En = SampleEvents.Sample_Neutrino_Energies(num_Events,E_min,E_max,power_law)
        
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
        power_law = 1.5
        #Make sure that power_law isn't equal to 1
        while power_law == 1:
            power_law = 1 + 2*rand.rand()
            
        print('power law',power_law)
        
        #Sample Energies
        En = SampleEvents.Sample_Neutrino_Energies(num_Events,E_min,E_max,power_law)
        
        n,bins,patches = plt.hist(En,50)
        midpoints = (bins[1:]+bins[:-1])/2
        
        #Calculate weighted differential dE at midpoints
        w_E = SampleEvents.weight_Energy(midpoints,E_min,E_max,power_law)
        
        #Check that the number of samples times dE gives uniform dist
        slope,intercept = np.polyfit(np.log(midpoints),np.log(n*w_E),1)
        self.assertAlmostEqual(slope, 0, delta = 0.1)
    '''
    #Test that the sampled event locations are
    #   uniform in volume if the sampling radius is the same
    def test_interaction_pos_sampling(self):
        R_Earth = 1
        R_min = 2e-6*R_Earth
        #R_Earth = 6378.1 * 1000* 100    #Radius of the Earth (cm)
        Y = np.array([0,0,R_Earth])
        Num_Events = int(1e5)
        R_max = 2*R_Earth * np.ones(Num_Events) #R_max so that whole earth covered
        
        #sample interaction locations
        x_vects = SampleEvents.Sample_Interaction_Locations(Num_Events,Y,R_max,R_min)
        
        #Find events within .1 R_Earth of center
        center_points = 0.25*R_Earth > np.sqrt(x_vects[:,0]**2 + x_vects[:,1]**2
                                              +x_vects[:,2]**2)
        
        print('center', sum(center_points))
        #Find events within .1 R_earth of (0,0,0.5 R_Earth)
        line_points = 0.25*R_Earth > np.sqrt(x_vects[:,0]**2 + x_vects[:,1]**2
                                              +(0.5*R_Earth - x_vects[:,2])**2)
        print('line', sum(line_points))
        
        #Check that there are roughly the same number of points in each area
        self.assertAlmostEqual(sum(center_points),sum(line_points),
                               delta = (np.sqrt(sum(center_points))+np.sqrt(sum(line_points))))
    
    
    #Test that weighting is done correctly, by summing all of the
    #   weights in the same volume with two different R_maxes
    def test_position_weighting(self):
        R_Earth = 1
        #R_Earth = 6378.1 * 1000* 100    #Radius of the Earth (cm)
        Y = np.array([0,0,R_Earth])
        Num_Events = int(1e5)
        R_max_1 = R_Earth
        R_max_2 = 2* R_Earth
        R_min = 2e-6*R_Earth
        
        #sample interaction locations
        x_vects_1 = SampleEvents.Sample_Interaction_Locations(Num_Events,Y,
                                                              R_max_1*np.ones(Num_Events),R_min)
        x_vects_2 = SampleEvents.Sample_Interaction_Locations(Num_Events,Y,
                                                              R_max_2*np.ones(Num_Events),R_min)
        
        #Find events within .1 R_earth of (0,0,0.5 R_Earth)
        first_points = 0.2*R_Earth > np.sqrt(x_vects_1[:,0]**2 + x_vects_1[:,1]**2
                                              +(0.5*R_Earth - x_vects_1[:,2])**2)
        
        first_total = sum(first_points) * SampleEvents.weight_positions(Y,R_max_1,R_min)
        
        second_points = 0.2*R_Earth > np.sqrt(x_vects_2[:,0]**2 + x_vects_2[:,1]**2
                                              +(0.5*R_Earth - x_vects_2[:,2])**2)
        
        second_total = sum(second_points)* SampleEvents.weight_positions(Y,R_max_2,R_min)
        
        self.assertAlmostEqual(first_total,second_total, delta = (np.sqrt(first_total)
                                                                  +np.sqrt(second_total)))
    '''    
    #Test that sampling in the exponential function does 
    #   give an expoential distribution
    def test_exp_pos_sampling(self):
        Num_Events = int(5e5)
        R_Earth = 1
        R_min = 2e-6 * R_Earth
        N_lambdas_1 = R_Earth*np.ones(Num_Events)
        
        
        Y1 = np.array([0,0,1e-6])
        x_vect_vals_1 = SampleEvents.Sample_Interaction_Location_Exp(Num_Events,Y1,R_min,N_lambdas_1)
        rs_1 = np.sqrt(x_vect_vals_1[:,0]**2 + x_vect_vals_1[:,1]**2 + x_vect_vals_1[:,2]**2)
        n,bins,patches = plt.hist(rs_1,40)
        midpoints = (bins[1:]+bins[:-1])/2
        slope,intercept = np.polyfit(midpoints,np.log(n),1)
        self.assertAlmostEqual(slope, -1/R_Earth,delta = 0.1)
        
        Y2 = np.array([0,0,0.99*R_Earth])
        x_vect_vals_2 = SampleEvents.Sample_Interaction_Location_Exp(Num_Events,Y2,R_min,N_lambdas_1)
        rs_2 = np.sqrt((x_vect_vals_2[:,0]-Y2[0])**2 + (x_vect_vals_2[:,1]-Y2[1])**2 \
                       + (x_vect_vals_2[:,2]-Y2[2])**2)
        print(rs_2[0:5])
        n,bins,patches = plt.hist(rs_2,40)
        midpoints = (bins[1:]+bins[:-1])/2
        slope,intercept = np.polyfit(midpoints,np.log(n),1)
        self.assertAlmostEqual(slope, -1/R_Earth,delta = 0.1)
        
        N_lambdas_2 = 0.5 * R_Earth * np.ones(Num_Events)
        x_vect_vals_3 = SampleEvents.Sample_Interaction_Location_Exp(Num_Events,Y2,R_min,N_lambdas_2)
        rs_3 = np.sqrt((x_vect_vals_3[:,0]-Y2[0])**2 + (x_vect_vals_3[:,1]-Y2[1])**2 \
                       + (x_vect_vals_3[:,2]-Y2[2])**2)
        n,bins,patches = plt.hist(rs_3,40)
        midpoints = (bins[1:]+bins[:-1])/2
        slope,intercept = np.polyfit(midpoints,np.log(n),1)
        self.assertAlmostEqual(slope, -1/(0.5*R_Earth),delta = 0.1)
        
    
    #Test that exponential sampling would give us the correct volume
    def test_exp_weight(self):
        Num_Events = int(5e5)
        R_Earth = 1
        R_min = 2e-6
        
        N_lambdas_1 = R_Earth*np.ones(Num_Events)
        Y1 = np.array([0,0,1e-6])
        x_vect_vals_1 = SampleEvents.Sample_Interaction_Location_Exp(Num_Events,Y1,R_min,N_lambdas_1)
        w_vs = SampleEvents.Weight_Positions_Exp(Y1,R_min,x_vect_vals_1,N_lambdas_1)
        self.assertAlmostEqual(sum(w_vs)/Num_Events, 1,delta = 0.1)
        
        N_lambdas_2 = R_Earth*np.ones(Num_Events)
        Y2 = np.array([0,0,0.99*R_Earth])
        x_vect_vals_2 = SampleEvents.Sample_Interaction_Location_Exp(Num_Events,Y2,R_min,N_lambdas_2)
        w_vs = SampleEvents.Weight_Positions_Exp(Y2,R_min,x_vect_vals_2,N_lambdas_2)
        self.assertAlmostEqual(sum(w_vs)/Num_Events, 1,delta = 0.1)
        
        N_lambdas_3 = 0.5*R_Earth*np.ones(Num_Events)
        Y3 = np.array([0,0,0.99*R_Earth])
        x_vect_vals_3 = SampleEvents.Sample_Interaction_Location_Exp(Num_Events,Y3,R_min,N_lambdas_3)
        w_vs = SampleEvents.Weight_Positions_Exp(Y3,R_min,x_vect_vals_3,N_lambdas_3)
        self.assertAlmostEqual(sum(w_vs)/Num_Events, 1,delta = 0.1)
        
    
    
    #Test that the selected entry position really does have the 
    #   proper scattering angle
    def test_entry_position(self):
        R_Earth = 1#6378.1 * 1000* 100    #Radius of the Earth (cm)
        Y = np.array([0,0,R_Earth])
        Xs = R_Earth/2 * rand.rand(10,3)
        cos_Thetas = 2*rand.rand(10) - 1
        Ws = SampleEvents.Sample_Neutrino_Entry_Position(Xs,Y,cos_Thetas)
        for k in range(10):
            W = Ws[k]
            X = Xs[k]
            cos_Theta = cos_Thetas[k]
            new_cos = np.dot(Y-X, X-W) /np.sqrt(np.dot(X-W,X-W) * np.dot(Y-X,Y-X))
            self.assertAlmostEqual(cos_Theta, new_cos, delta = 1e-6)
    

class Testing_Detector_Functions(unittest.TestCase):
    #Test that a completely forward (backward) scattered photon
    #   in the rest frame is still forward (backward) in the lab frame
    #Also test Energies are correct
    def test_E_gamma_and_zeta(self):
        En = 1 #Lepton Energy in GeV
        mn = 0.1 #Lepton mass in GeV
        f_cos_zeta_prime = 1
        b_cos_zeta_prime = -1
        
        f_zeta, f_E_gamma = DetectorModule.Calc_Zetas_and_Energies(f_cos_zeta_prime,En,mn)
        b_zeta, b_E_gamma = DetectorModule.Calc_Zetas_and_Energies(b_cos_zeta_prime, En, mn)
        
        self.assertEqual(f_zeta,0)
        self.assertEqual(b_zeta,pi)
        self.assertEqual(f_E_gamma, 0.5 * En * (1 + np.sqrt(1 - mn**2/En**2)))
        self.assertEqual(b_E_gamma, 0.5 * En * (1 - np.sqrt(1 - mn**2/En**2)))
    
    #Test that we get the proper angle relative to the detector when we
    #   have the interaction directly below the detector, and 0 as the
    #   photon scattering angle
    def test_phi_det(self):
        R_Earth = 1   #Radius of the Earth (cm)
        Y = np.array([0,0,R_Earth])
        X = np.array([[0,1e-6,1e-6]])
        
        En = np.array([1]) #Lepton Energy in GeV
        mn = 0.1 #Lepton mass in GeV
        cos_zeta_prime = np.array([1])
        
        zeta, E_gamma = DetectorModule.Calc_Zetas_and_Energies(cos_zeta_prime,En,mn)
        
        cos_phi_det = DetectorModule.Calc_cos_phi_det(Y,X,zeta)
        
        self.assertAlmostEqual(cos_phi_det, -1, delta = 1e-6)

if __name__ == "__main__":
    unittest.main()
