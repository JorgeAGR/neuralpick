# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 12:25:30 2020

@author: jorge
"""

import os
import obspy
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from time import time as clock

class Scanner(object):

    def __init__(self, file_dir, ModelType, phase, begin_pred, end_pred, relevant_preds):
        write_path = 'scans/'
        if write_path.rstrip('/') not in os.listdir():
            os.mkdir(write_path)
        
        self.resample_Hz = ModelType.sample_rate
        self.time_window = ModelType.window_before + ModelType.window_after
        self.phase = phase
        
        ModelType.load_Model()
        pred_model = ModelType.model
        self.n_preds = self.time_window * self.resample_Hz # Maximum number of times the peak could be found, from sliding the window
        
        files = np.sort(os.listdir(file_dir))
        extension = '.{}'.format(files[0].split('.')[-1])
        files = np.sort([f.rstrip(extension) for f in files if extension in f])
        gen_whitespace = lambda x: ' '*len(x)
        pred_time = 0
        
        with open(write_path + file_dir.split('/')[-2] + '_{}_{}_scan.csv'.format(begin_pred, end_pred), 'w+') as pred_csv:
            print_cols = lambda x: 'pred{0},err{0},amp{0},qual{0},'.format(x)
            header_string = 'file,gcarc,'
            for i in range(relevant_preds):
                header_string += print_cols(i)
            header_string = header_string.rstrip(',')
            print(header_string, file=pred_csv)
        
        for f, sac_file in enumerate(files):
            #print('File', f+1, '/', len(files),'...', end=' ')
            print_string = 'File {} / {}... Est. Time per Prediction: {:.2f} sec'.format(f+1, len(files), pred_time)
            print('\r'+print_string, end=gen_whitespace(print_string))
            tick = clock()
            gcarc, results = self.find_Precursors(file_dir, sac_file, extension, pred_model, 
                                      relevant_preds, begin_pred, end_pred)
            tock = clock()
            if f == 0:
                pred_time = tock-tick
            with open(write_path + file_dir.split('/')[-2] + '_{}_{}_scan.csv'.format(begin_pred, end_pred), 'a') as pred_csv:
                print('{},{}'.format(sac_file, gcarc), end=',', file=pred_csv)
                for i in range(relevant_preds-1):
                    print(results[i], end=',', file=pred_csv)
                print(results[-1], file=pred_csv)

    def cut_Window(self, cross_sec, times, t_i, t_f):
        # This should be redefined to find closest initial point in time
        # grid and then defnining points after that.
        init = int(np.round(t_i*self.resample_Hz))
        end = int(np.round(t_f*self.resample_Hz))
        
        return cross_sec[init:end]
    
    def shift_Max(self, seis, pred_var):
        data = seis.data
        time = seis.times()
        arrival = 0
        new_arrival = seis.stats.sac[pred_var]
        while (new_arrival - arrival) != 0:
            arrival = new_arrival
            # Again, same as above
            init = np.where(time > (arrival - 1))[0][0]
            end = np.where(time > (arrival + 1))[0][0]
            
            amp_max = np.argmax(np.abs(data[init:end]))
            time_ind = np.arange(init, end, 1)[amp_max]
            
            new_arrival = time[time_ind]
        return arrival
    
    def scan(self, seis, times, time_i_grid, time_f_grid, shift, model, negative=False):
        window_preds = np.zeros(len(time_i_grid))
        for i, t_i, t_f in zip(range(len(time_i_grid)), time_i_grid, time_f_grid):
            seis_window = self.cut_Window(seis, times, t_i, t_f) * (-1)**negative
            seis_window = seis_window / np.abs(seis_window).max()
            # Take the absolute value of the prediction to remove any wonky behavior in finding the max
            # Doesn't matter since they are bad predictions anyways
            window_preds[i] += np.abs(model.predict(seis_window.reshape(1, len(seis_window), 1))[0][0]) + t_i
        return window_preds
    
    def find_Precursors(self, file_dir, sac_file, extension, model, relevant_preds, pred_init_t, pred_end_t):
        seis = obspy.read(file_dir+sac_file+extension)
        seis = seis[0].resample(self.resample_Hz)
        times = seis.times()
        shift = -seis.stats.sac.b
        
        phases_in_seis = [seis.stats.sac[k].rstrip(' ').rstrip('ap') for k in seis.stats.sac.keys() if 'kt' in k]
        phases_headers = [k.lstrip('k') for k in seis.stats.sac.keys() if 'kt' in k]
        phase_var = dict(zip(phases_in_seis, phases_headers))[self.phase]
        
        '''
        # This should maybe be changed to closest time point in time grid, for
        # use in other sample frequencies. Then the initial and final grids can
        # be determined from the number of points after in the seis.time() array.
        '''
        if shift < np.abs(pred_init_t):
            pred_init_t = -shift
        begin_time = np.round(seis.stats.sac[phase_var] + pred_init_t + shift, decimals=1)
        end_time = np.round(seis.stats.sac[phase_var] + pred_end_t + shift, decimals=1)
        
        time_step = 1/self.resample_Hz
        time_i_grid = np.arange(begin_time, end_time - self.time_window + time_step, time_step)
        time_f_grid = np.arange(begin_time + self.time_window, end_time + time_step, time_step)
        window_preds = np.zeros(len(time_i_grid))
        window_preds = self.scan(seis, times, time_i_grid, time_f_grid, shift, model)
        
        dbscan = DBSCAN(eps=time_step/2, min_samples=2)
        dbscan.fit(window_preds.reshape(-1,1))
        clusters, counts = np.unique(dbscan.labels_, return_counts=True)
        if -1 in clusters:
            clusters = clusters[1:]
            counts = counts[1:]
        
        discont_ind = np.argsort(counts)[-relevant_preds:]
        clusters = clusters[discont_ind]
        counts = counts[discont_ind]
        arrivals = np.zeros(relevant_preds)
        arrivals_err = np.zeros(relevant_preds)
        arrivals_amps = np.zeros(relevant_preds)
        arrivals_quality = np.zeros(relevant_preds)
        for i, c in enumerate(clusters):
            arrivals[i] = np.mean(window_preds[dbscan.labels_ == c])
            arrivals_err[i] = np.std(window_preds[dbscan.labels_ == c])
            arrivals_quality[i] = counts[i] / self.n_preds
            initamp = np.where(times < arrivals[i])[0][-1]
            arrivals_amps[i] = seis.data[initamp:initamp+2].max()
        arrivals = arrivals - shift
        
        make_string = lambda x: '{},{},{},{}'.format(arrivals[x],arrivals_err[x],arrivals_amps[x],arrivals_quality[x])
        result_strings = list(map(make_string, range(relevant_preds)))
        return seis.stats.sac.gcarc, result_strings