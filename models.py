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
    Calculates spectral index for a given model using the ratio of hard band and
    softband fluxes.

    Parameters
    ----------
    model_specific_flux : function
        function that givcs specific flux of the function.
    t_list : array 
        Array of times for which flux is to be calculated.
    E_s : Float, optional
        Soft band energy. The default is 0.3.
    E_h : float, optional
        Hard band energy. The default is 10..
    args : list, optional
        optional arguments to be passed to model_specific_flux. The default is [].

    Returns
    -------
    spectral_index : numpy array
        array of spectral index.

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
    .Equations and symbols are taken from
    Shao et. al 2008 (DOI: 10.1086/527047).
    Written to turn it into low level C callable.

    Parameters
    ----------
    n : integer
        Number of arguments i.e len(args)
    args : array of parameters
        args = [a, E, t, R, s, q, z, S0, beta] where 
        S(E) = fluence of source at Energy E at E
        beta = spectral index of power law source function
    

    Returns
    -------
    dF : float
        Integrand to be integrated using low level callble in scipy integrate.

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
    Finds model specific flux for dust model.Equations and symbols are taken from
    Shao et. al 2008 (DOI: 10.1086/527047).

    Parameters
    ----------
    E : float
        Energy (in keV) at which flux is to be calculated.
    t_list : list
        List of times for which it is to be calculated.
    a_m : float
        Lower cut off for grain size (in um).
    a_M : float
        Upper cut off for grain size (in um).
    R_pc : floar
        Distance to dust screen in pc.
    s : float
        power index for tau dependence on energy.
    q : float
        power index for dust size distribiution.
    tau0 : float
        Normalisation for optical depth.
    z : float
        Redshift of host galaxy.
    source_fluence_E : float
        Source fluence at energy E in ergs/keV.
    tol : floar, optional
        epabs for scipy.integrate . The default is 1.49e-8.
    lim : integer, optional
        lim for scipy.integrate.. The default is 100.

    Returns
    -------
    F_dust : array
        An array of flux (in ergs/keV) with same length as t_list.

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
    Integrand written for fast running in numba for total flux for dust model 
    with power law source function.Arguments are
    n = number of args and args = [a, E, t, R, s, q, z, S0, beta] where
    .Equations and symbols are taken from
    Shao et. al 2008 (DOI: 10.1086/527047).
    Written to turn it into low level C callable.

    Parameters
    ----------
    n : integer
        Number of arguments i.e len(args)
    args : array of parameters
        args = [a, E, t, R, s, q, z, S0, beta] where 
        S0 = Normalisation of power law source function
        beta = spectral index of power law source function
    

    Returns
    -------
    dF : float
        Integrand to be integrated using low level callble in scipy integrate.

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
    E in ergs/keV. Returns the answer in ergs and error.

    Parameters
    ----------
    t_list : list
        List of times for which it is to be calculated.
    E_m : float
        Lower limit for energy integration in keV.
    E_M : float
        Upper limit for energy integration in keV.
    a_m : float
        Lower cut off for grain size (in um).
    a_M : float
        Upper cut off for grain size (in um).
    R_pc : floar
        Distance to dust screen in pc.
    s : float
        power index for tau dependence on energy.
    q : float
        power index for dust size distribiution.
    tau0 : float
        Normalisation for optical depth.
    z : float
        Redshift of host galaxy.
    S0 : float
        Normalisation of power law source function (in ergs/keV).
    beta : float
        Power of energy in power law source function
    tol : floar, optional
        epabs for scipy.integrate . The default is 1.49e-8.
    lim : integer, optional
        lim for scipy.integrate.. The default is 100.

    Returns
    -------
    F_dust : array
        An array of flux (in ergs) with same length as t_list.

    """
    # Need to check the dimensions of output
    F_dust = 0.1*tau0*np.array([integrate.nquad(ctype_total_f_PL,
                                                        ((a_m, a_M), (E_m, E_M)),
                                                        args=(t_i, R_pc, s, q, z, S0, beta)
                                                        ,opts={"epsabs":tol,"limit":lim})
                                for t_i in t_list])
    return F_dust
    