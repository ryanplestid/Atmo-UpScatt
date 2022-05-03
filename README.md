# Atmo-UpScatt

Python based code for performing a Monte Carlo simulation in which atmospheric neutrinos upscatter into heavy neutral leptons (HNLs). Some modules are designed to be model independent, while some are specific to the HNL dipole or mass-mixing portal. The simulation preferentially samples the energy, interaction location, scattering angle, and neutrino production position, and then weights the events accordingly.

Along with an overall rate of HNL events in a detector, kinematic properties about the decay products can also be evaluated.

## Requirements:
-- numpy

-- scipy

-- matplotlib

-- nuflux

## Code Included:

### "DetectorModule.py"

Module to calculate the observed properties of the photon from the HNL decay.
Made with the dipole coupling portal in mind

### "MainIntegration.py"

Module that performs the Monte Carlo integration for the HNL dipole portal

### "MassMixingMainIntegration.py"
Module that performs the Monte Carlo integration for the HNL Mass Mixing portal

### "Mass_Mixing_Reweight.py"
Code to reweight a Monte Carlo integration for the HNL mass-mixing portal with a new HNL mass and coupling strength

### "SampleEvents.py"
Code to sample the properties of our events at the beginning of the Monte Carlo simulation

### "atmoNuIntensity.py"
Module to determine the proper neutrino flux for each event

### "dipoleModule.py"
Module to calculate the decay length of the HNL and the scattering cross sections specific to the dipole model.

### "earthComp.py"
Module to calculate details about the composition of Earth

### "formFactorFit.py"
Module to compute the form factors for nuclei used to calculate contributions of coherent and incoherent scattering.

### "massMixingModel.py"
Module to calculate the decay length of the HNL and the scattering cross sections specific to the mass-mixing model

### "oscillations.py"
Module to calculate the flavor oscillation of the neutrino along a linear path

### "reweightModule.py"
Code to reweight a Monte Carlo integration for the HNL dipole portal with a new HNL mass and dipole coupling.

## Sample Scripts

### "MonteCarloExample.py"
Code to run the Monte Carlo simulation and then use the ReIntegrate function to recalculate the rate of HNL events. Also included is an example on calculating the angular and energetic data of the decay photons from the ReIntegrate function

### "BoundExample.py"
Example showing how to calculate the bounds in parameter space given a desired number of visible HNL events. This method calculates a lower bound and a right bound and pieces them together.  This works well when the bounds are shaped similarly to the dipole portal bounds presented in the paper.
  
