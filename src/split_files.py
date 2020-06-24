#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 02:03:56 2020

@author: jorgeagr
"""

import os
import argparse
import subprocess

parser = argparse.ArgumentParser(description='Predict precursor arrivals in vespagram cross-sectional data.')
parser.add_argument('file_dir', help='SAC files directory.', type=str)
parser.add_argument('partitions', help='Number of partitions to create', type=int)
args = parser.parse_args()

file_dir = args.file_dir
partitions = args.partitions
if file_dir[-1] != '/':
    file_dir += '/'

files = list(map(lambda x: file_dir+x, sorted(os.listdir(file_dir))))

part_size = len(files) // partitions

for i in range(partitions):
    partdir = '{}{}'.format(file_dir.split('/')[-2], i)
    os.mkdir(file_dir + partdir)#'{}{}/'.format(file_dir.split('/')[-2],i))
    if i != partitions-1:
        part_files = files[part_size*i:part_size*(i+1)]
    else:
        part_files = files[part_size*i:]
    part_files.insert(0, 'mv')
    part_files.append('-t')
    part_files.append(file_dir + partdir)#'{}{}/'.format(i))
    subprocess.call(part_files)
