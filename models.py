#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 03:54:56 2021

@author: asif
"""
import numpy as np
from scipy import integrate,LowLevelCallable
from numba import cfunc, types

c=299792458.# velocity of light in m/s
h=6.626e-34


kev=1.6022e-16
pc=3.086e+16
um=1e-6

norm=1/9
conversion_factor=413560.0 # Converting from erg cm^{-2}/keV to Jy

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
