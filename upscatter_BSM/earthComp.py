'''
Module to calculate details about the composition of Earth

functions:
    which_layer: Specifies the layer of the Earth given the radius
    rho_PREM: Compute the density of Earth at a given radius
        according to the Preliminary Reference Earth Model
    rho_Const: Returns a constant density as long as the specified
        radius is within Earth
    n_density: Calculates the number density of each element at
        a specified radius
    gen_1d_ne: Calculate a 1 dimensional profile of the number
        density of electrons between two points
        
dictionaries:
    molar_mass: Molar masses of Earth elements
    atomic_number: Atomic number of Earth elements
    neutron_number: Neutron number for Earth elements
            
'''

import numpy as np
import warnings

from  numpy import sin,cos,exp,sqrt,pi

# in g/mol
molar_mass={"O" : 15.9994, "Si": 28.0855, "Al": 26.9815, \
            "Fe": 55.845, "Ca": 40.078, "Na": 22.9897, \
            "K" : 39.098, "Mg": 24.305,  "S": 32.065, \
            "Ni": 58.6934}
 
atomic_number={"O" : 8, "Si": 14, "Al": 13, \
               "Fe": 26, "Ca": 20, "Na": 11, \
               "K" : 19, "Mg": 12,  "S": 16, \
               "Ni": 28}

neutron_number={"O" : 8, "Si": 14, "Al": 14, \
               "Fe": 30, "Ca": 20, "Na": 12, \
               "K" : 20, "Mg": 12,  "S": 16, \
                "Ni": 30}



n_avagadro=6.022*1E23


def main():
    r0=0.99
    n_e=0

    X1=np.asarray( [ 0.995, 0, 0] )
    X2=np.asarray( [ 1, 0, 0])

    s,n_e= gen_1d_ne(X1,X2)

    #    print(n_e)
    print(n_density(r0)["e"]/n_avagadro , n_density(0)["e"]/n_avagadro  )
    return 0




def which_layer(r):
    '''
    Which layer of composition does the current radius
    belong to
    
    args:
          r: Currrent radius in units where r_Earth =1 

    returns: 
             layer_string  which describes current layer
    '''
    
    r_earth_in_km = 6371
    crust_thickness=50
    
    r_inner_core  = 1217.5/r_earth_in_km
    r_outer_core  = 3479.5/r_earth_in_km
    r_disc_660    = (r_earth_in_km -660)/r_earth_in_km
    r_disc_410    = (r_earth_in_km -410)/r_earth_in_km
    r_crust       = (r_earth_in_km-crust_thickness)/r_earth_in_km


    if r<= r_inner_core:
        return("Inner Core")
    elif r<= r_outer_core:
        return("Outer Core")
    elif r<= r_disc_660:
        return("Lower Mantle")
    elif r<= r_crust:
        return("Upper Mantle")
    elif r<= 1:
        return("Crust")
    else:
        return("Space")

 
def rho_PREM(r):
    '''
     Function that computes density as a function of r 
     Based on dziewonski et. al (1981) Table IV

    args: 
         r: Real, valued number representing the radius in
            units of R_earth =1. r>1 returns 0

    returns; 
            rho: Real,  Density in g/cm^3 at r 
    '''
    # TABLE I, pg 308 of Dziewonski et. al. 1981 
    r_in_km=6371*r


    if r_in_km<1221.5:
        return(13.0885-8.8381*r**2)
        
    elif r_in_km < 3480:
        return(12.5815-1.2638*r-3.6426*r**2-5.5281*r**3)

    elif r_in_km < 3630:
        return(7.9565-6.4761*r+5.5283*r**2-3.0807*r**3)

    elif r_in_km < 5600:
        return(7.9565-6.4761*r+5.5283*r**2-3.0807*r**3)

    elif r_in_km < 5701:
        return(7.9565-6.4761*r+5.5283*r**2-3.0807*r**3)

    elif r_in_km < 5771:
        return(5.3197-1.486*r)

    elif r_in_km < 5971:
        return(11.2494-8.0298*r)

    elif r_in_km < 6151:
        return(7.1089-3.8045*r)

    elif r_in_km < 6291:
        return(2.6910+0.6924*r)

    elif r_in_km < 6346.6:
        return(2.6910+0.6924*r)

    elif r_in_km < 6356.0:
        return(2.900)

    elif r_in_km < 6368.0:
        return(2.600)

    elif r_in_km < 6371.0:
        # This differs from the PREM
        # We just take "rock" everywhere
        # PREM sets this "Ocean" layer's
        # desnity to 1.020
        return(2.600)

    else:
        return(0)


def rho_Const(r):
    '''
     Function that computes density as a function of r 
     Based on dziewonski et. al (1981) Table IV

    args: 
         r: Real, valued number representing the radius in
            units of R_earth =1. r>1 returns 0

    returns; 
            rho: Real,  Density in g/cm^3 at r 
    '''
    # TABLE I, pg 308 of Dziewonski et. al. 1981 
    r_in_km=6371*r

    if r_in_km < 6371.0:
        # This differs from the PREM
        # We just take "rock" everywhere
        # PREM sets this "Ocean" layer's
        # desnity to 1.020
        return(5)

    else:
        return(0)


def n_density(r, dens=rho_PREM):
    '''
     Function that computes density as a function of r 
     Based on dziewonski et. al (1981) Table IV

    args: 
         r: Real, valued number representing the radius in
            units of R_earth =1. r>1 returns 0

    returns; 
            n_dens: dictionary of number densities relevant for 
                    specific locations in units of N/cm^3
    '''

    if which_layer(r)=="Space":
        return({})



    elif which_layer(r)=="Crust":
        wgt_frac={"O" : 0.466, "Si": 0.2772 , "Al": 0.0813,\
                  "Fe": 0.0505, "Ca": 0.0365 , "Na": 0.0275,\
                  "K" : 0.0258, "Mg": 0.0208}

    #McDonough, W. F. (2014). Compositional Model for the
    #Earth’s Core. Treatise on Geochemistry, 559–577.
    #doi:10.1016/b978-0-08-095975-7.00215-1
    # Table  5 
    elif which_layer(r)=="Upper Mantle":
        wgt_frac={"O": 0.44, "Si": 0.21 , "Al": 0.0253,\
                  "Fe": 0.0626, "Ca": 0.025, "Mg": 0.228}

    elif which_layer(r)=="Lower Mantle":        
        wgt_frac={"O": 0.44, "Si": 0.21 , "Al": 0.0253,\
                  "Fe": 0.0626, "Ca": 0.025, "Mg": 0.228}


    elif which_layer(r)=="Inner Core":
        wgt_frac={"S": 0.019, "Si": 0.06 ,"Fe": 0.855,\
                  "Ni": 0.052} 

    elif which_layer(r)=="Outer Core":
        wgt_frac={"S": 0.019, "Si": 0.06,"Fe": 0.855,\
                  "Ni": 0.052} 


    else:
        warnings.warn("Layer did not exist")
        wgt_frac={}


    assert(sum(wgt_frac.values()) < 1)
    assert(sum(wgt_frac.values()) > 0.975)
    n_dens={}
    
    for key in wgt_frac:
        n_dens[key]= dens(r)*wgt_frac[key]/molar_mass[key]*n_avagadro

    n_e=0
    for key in n_dens:
        n_e = n_e + n_dens[key]*atomic_number[key]

    # we systematically undercount elements above
    # This corrects for the undercounting of species
    n_dens["e"]=n_e/( sum(wgt_frac.values()) )

    return(n_dens)
    
def gen_1d_ne( X1, X2, n_points=100):
    '''
    Generates a 1 dimensional density profile of electron number density
    given two position vectors, X1, X2. 

    args: 

          X1: 3-dim array of coords in units of r_earth =1 

          X2: 3-dim array of coords in units of r_earth =1 
    
          n_points=100 [default] number of points in generated 1d array

    returns: 

           1d array of n_e vs x in units of N/cm^3

    '''
    diff_vec=X2-X1
    dist    =sqrt(diff_vec@diff_vec)
    unit_vec=diff_vec/dist

    s=np.linspace(0,dist,n_points)

    n_e=[]
    for i in range(n_points):

        X=X1+s[i]*unit_vec
        r=sqrt(X@X)
        
        n_e.append(n_density(r)["e"])

    n_e=np.asarray(n_e)

    return(s, n_e)
    

    



if __name__ == "__main__":
    main()
