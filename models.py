#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 03:54:56 2021

@author: asif
"""
import numpy as np
from scipy import integrate,LowLevelCallable
from numba import cfunc, types

c=299792458.  # velocity of light in m/s
h=6.626e-34


kev=1.6022e-16
pc=3.086e+16
um=1e-6

# conversion_factor=413560.0 # Converting from erg cm^{-2}/keV to Jy


def calc_spectral_index(model_specific_flux, t_list, E_s = 0.3, E_h = 10., args=[]):
    """
    Takes in a function for specific_flux(E, t, *args) and returns spectral
    idex by taking ration between fluxes at E_m and E_M
    """  

    F_h = model_specific_flux(E_h, t_list, *args)
    F_s = model_specific_flux(E_s, t_list, *args)  
    spectral_index= np.log(F_s/F_h)/np.log(E_h/E_s)
    return spectral_index


def PL_model(E,t,alpha,beta,norm):
    return norm*np.power(E,-beta)*np.power(t,-alpha)

c_sig = types.double(types.intc, types.CPointer(types.double))

@cfunc(c_sig)
def specific_flux_integrand(n,args):
    """
    Integrand written for fast running in numba for specific flux.Arguments are
    n = number of args and args = [a, E, t, R, s, q, z, S(E)] where
    S(E) = fluence of source at Energy E. Equations are taken from
    Shao et. al 2008 (DOI: 10.1086/527047).
    Written to turn it into low level C callable.
    """
    theta=np.sqrt(2*299792458*args[2]/((1+args[6])*args[3]*3.086e+16))    
    
    x=(2*np.pi/(6.626e-34*299792458))*(1+args[6])*theta*args[1]*1.6022e-16*args[0]*1e-6
    
    tau_a=np.power(args[1]*(1+args[6]),-args[4])*np.power(args[0]/0.1,4.-args[5])
    temp=(np.sin(x)/x**2) - (np.cos(x)/x)
    tau_theta=np.power(temp,2)
    tau=tau_a*tau_theta
    
    S = args[7]
    dF=S*tau/args[2]
    return dF

ctype_specific_f = LowLevelCallable(specific_flux_integrand.ctypes)

def calc_dust_specific_flux(E, t_list, a_m, a_M, R_pc, s, q, tau0, z, 
                            source_fluence_E,  tol = 1.49e-8, lim = 100):
    """
    Calculates specific flux as per dust model with given parameters where
    E(keV), t(s), a_m and a_M are lower and upper cut offs of grain sizes in
    um, R_pc is the distance to dust layer in pc, s and q are as defined in
    Shao et. al 2008, z = redshift, source_fluence_E = source fluence at energy
    E in ergs/keV. t_list is expected as a iterable.
    Returns the answer in ergs/keV and error.
    """
    
    F_dust = 0.1*tau0*np.array([integrate.quad(ctype_specific_f, a_m, a_M,
                                              args=(E, t_i, R_pc, s,q, z, source_fluence_E),
                                              epsabs=tol,limit=lim
                                              )
                                for t_i in t_list])
    return F_dust

@cfunc(c_sig)
def total_flux_integrand_PL(n,args):
    """
    Integrand written for fast running in numba for specific flux.Arguments are
    n = number of args and args = [a, E, t, R, s, q, z, S0, beta] where 
    S(E) = fluence of source at Energy E. Equations are taken from 
    Shao et. al 2008 (DOI: 10.1086/527047).
    Written to turn it into low level C callable.
    """
    theta=np.sqrt(2*299792458*args[2]/((1+args[6])*args[3]*3.086e+16))    
    
    x=(2*np.pi/(6.626e-34*299792458))*(1+args[6])*theta*args[1]*1.6022e-16*args[0]*1e-6
    
    tau_a=np.power(args[1]*(1+args[6]),-args[4])*np.power(args[0]/0.1,4.-args[5])
    temp=(np.sin(x)/x**2) - (np.cos(x)/x)
    tau_theta=np.power(temp,2)
    tau=tau_a*tau_theta
    
    S = args[7]*np.power(args[1],-args[8])
    dF=S*tau/args[2]
    return dF

ctype_total_f_PL = LowLevelCallable(total_flux_integrand_PL.ctypes)

def calc_dust_total_flux_PL(t_list, E_m, E_M, a_m, a_M, R_pc, s, q, tau0, z, S0, 
                         beta, tol = 1.49e-8, lim = 100):
    """
    Calculates total flux as per dust model with power law source function.
    given parameters where t(s), E_m and E_M are lower and upper cut offs of 
    the band we are observing a_m and a_M are lower and upper cut offs of grain sizes in 
    um, R_pc is the distance to dust layer in pc, s and q are as defined in 
    Shao et. al 2008, z = redshift, source_fluence_E = source fluence at energy
    E in ergs/keV. Returns the answer in ergs/keV and error.
    """
    # Need to check the dimensions of output
    F_dust = 0.1*tau0*np.array([integrate.nquad(ctype_total_f_PL,
                                                        ((a_m, a_M), (E_m, E_M)),
                                                        args=(t_i, R_pc, s, q, z, S0, beta)
                                                        ,opts={"epsabs":tol,"limit":lim})
                                for t_i in t_list])
    return F_dust
    