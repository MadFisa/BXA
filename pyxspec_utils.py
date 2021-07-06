#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 20:55:18 2021

@author: asif
"""

import xspec as xp
# import numpy as np
# from scipy import integrate,LowLevelCallable
from models import *
from tqdm import tqdm
from astropy.io import fits

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
