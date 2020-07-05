# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 20:44:04 2020

@author: jorge
"""

import os
import obspy
import numpy as np
from scipy.interpolate import interp1d
from sklearn.cluster import DBSCAN
from time import time as clock

class QualityChecker(object):
    
    def __init__(self, file_dir, ModelType, phase):
        self.resample_Hz = ModelType.sample_rate
        self.time_window = ModelType.window_before + ModelType.window_after
        
        model = ModelType.load_Model()
        
        files = np.sort([f for f in os.listdir(file_dir) if '.s_fil' in f])
        gen_whitespace = lambda x: ' '*len(x)
        
        print('\nQuality checking around', phase, 'phase for', len(files), 'files.')
        pred_time = 0
        for f, seis_file in enumerate(files):
            print_string = 'File {} / {}... Est. Time per Prediction: {:.2f} sec'.format(f+1, len(files), pred_time)
            print('\r'+print_string, end=gen_whitespace(print_string))
            try:
                tick = clock()
                self.pick_Phase(file_dir, seis_file, phase, model)
                tock = clock()
                if f == 0:
                    pred_time = tock-tick
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exception:
                if 'bad.log' not in os.listdir(file_dir + 'picked/'*-(~overwrite)):
                    self.write_Exception(file_dir, f, seis_file, exception, mode='w+')
                else:
                    self.write_Exception(file_dir, f, seis_file, exception)
        print('\nSeismograms picked. Bon appetit!')
        
        return
    
    def quality_Check(file_dir, seis_file, phase, model):
        