#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 15:39:46 2021

@author: asif
"""
import numpy as np


def read_qdp(qdp_file_name):
    """The function reads a qdp file ignoring NO and !. Returns Dictionary with names and list of data"""
    with open(qdp_file_name,'r') as reader:
        lines=reader.readlines()
        modes=[]
        data=[]
        b=iter(lines[9:])
        data=[]
        block=[]
        for line in b:
            if  line[0].isalpha():
                data.append(block)
                block=[]            
                line=next(b)            
                modes.append(line.strip().split(' ')[1])
                line=next(b)
                line=next(b)
            block.append(line.strip().split('\t'))    
            # print(line)
        data.append(block)
        data.pop(0)
        return [modes,data]

modes,data=read_qdp("XRT unabsorbed flux light curves_10keV.qdp")