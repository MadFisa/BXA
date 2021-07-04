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
        data_dict = {modes_i:np.array(data_i,dtype=np.float32) for modes_i,data_i in zip(modes,data)} 
        return data_dict

def data_to_dict(data):
    return {'data':data[:,0], 'p_err':data[:,1], 'n_err':data[:,2]}

def split_block(block,labels = ['time', 'value']):
    """ Takes in a block numpy of the form [[label1_data,label1_+ve_err,label1_-ver_err]
                                            [label2,label2_+ver_err,label2_-ver_err]]
    and split into a dictionary of type data['label']['data'\'p_err'\ 'n_err']
    """
    data_dict = {}
    for idx,labels_i in enumerate(labels):
        temp = block[:, 3*idx : 3*(idx+1)]
        data_dict[labels_i] = data_to_dict(temp)
    return data_dict

    