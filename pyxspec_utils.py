#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 20:55:18 2021

@author: asif
"""

import xspec as xp
import numpy as np
from tqdm import tqdm
from astropy.io import fits
from scipy import integrate,LowLevelCallable
from numba import cfunc, types

tol = 1.49e-8
lim = 100

def add_xflt(spec_list):
    """
    Function to add time information in a way accessible to PyXspec. It can only access 
    xflt field. This funtion will take information from spectrum files and adds a the info
    to xlft fields. The input is a list of spectrum files.
    """
    for spec in spec_list:
        with fits.open(spec, mode='update') as spec_fits:
            print(f'Reading {spec}')
            t_start = spec_fits[1].header["TSTART"]
            t_stop = spec_fits[1].header["TSTOP"]
            t_trig = spec_fits[1].header["TRIGTIME"]
            spec_t_start = t_start-t_trig
            spec_t_stop = t_stop-t_trig
            spec_fits[1].header["XFLT0001"] = spec_t_start
            spec_fits[1].header["XFLT0002"] = spec_t_stop
            print(f'Updated {spec} with t_start = {spec_t_start} \
                                        t_stop = {spec_t_stop}')
            spec_fits.flush()

c_sig = types.double(types.intc, types.CPointer(types.double))

@cfunc(c_sig)
def pyXspec_integrand_PL(n,args):
    """
    Integrand for pyXspec.Arguments are
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

ctype_pyXspec_f_PL = LowLevelCallable(pyXspec_integrand_PL.ctypes)

def dust_PL (engs,params,flux,flux_err,spec_num):
    """
    XSPEC dust model with power law source function  given parameters 
    params = [a_m, a_M, R_pc, s, q, tau0, z, S0, 
                         beta, tol = 1.49e-8, lim = 100]
    where E_m and E_M are lower and upper cut offs of 
    the band we are observing a_m and a_M are lower and upper cut offs of grain sizes in 
    um, R_pc is the distance to dust layer in pc, s and q are as defined in 
    Shao et. al 2008, z = redshift, source_fluence_E = source fluence at energy
    E in ergs/keV. t_l and t_u is the lower and upper limit of 
    the time interval of spectrum which should be
    written as xflt field in FITS file.
    Returns the answer in ergs/keV and error.
    """
    a_m, a_M, R_pc, s, q, tau0, z, S0, beta , nm= params
    n = len(engs)
    x1,x2=xp.AllData(spec_num).xflt
    t_l=x1[1]
    t_u=x2[1]
    for i in range(n - 1):
        flux[i] = 0.1*tau0*integrate.nquad(ctype_pyXspec_f_PL, 
                                        ((a_m,a_M), (engs[i],engs[i+1]),(t_l,t_u)),
                                        args = (R_pc, s, q, z, S0, beta))[0]

def add_dustPL_to_Xspec():
    #               par_name  units default hard_min soft_min soft_max hard_max fit_delta
    Dust_ModelInfo=("LowerDust \"\" 0.025 0.0001 0.001 0.3 1 0.01",
                "UpperDust \"\" 0.25 0.0001 0.01 0.8 2 0.01",
                "DustDistance \"\" 10 0.5 1 500 5000 0.1",
                "s \"\" 2 -5 0 6 10 0.1",
                "q \"\" 4.5 0 2.5 6 8 0.1",
                "tau0 \"\" 1 1e-5 1e-3 1e3 1e5 1e1",
                "z \"\" 0 0 0 10 100 0.1",
                "S0 \"\" 1e-6 1e-15 1e-8 1e1 1e3 1e1",
                "beta \"\" 2 0 0.5 5 10 0.1")

    xp.AllModels.addPyMod(dust_PL, Dust_ModelInfo, 'add')