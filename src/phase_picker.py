#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 17:48:13 2020

@author: jorgeagr
"""

import os
import obspy
import numpy as np
from scipy.interpolate import interp1d
from sklearn.cluster import DBSCAN
from time import time as clock

class Picker(object):
    
    def __init__(self, file_dir, ModelType, phase, overwrite=False):
        self.resample_Hz = ModelType.sample_rate
        self.time_window = ModelType.window_before + ModelType.window_after
        self.max_preds = self.time_window * self.resample_Hz # Maximum number of times the peak could be found, from sliding the window
        self.overwrite = overwrite
        
        ModelType.load_Model()
        
        model = ModelType.model
        
        files = np.sort([f for f in os.listdir(file_dir) if '.s_fil' in f])
        gen_whitespace = lambda x: ' '*len(x)
        
        if not overwrite:
            if 'picked' not in os.listdir(file_dir):
                os.mkdir(file_dir + 'picked/')
        
        print('\nPicking for', phase, 'phase in', len(files), 'files.')
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
                if '{}_bad.log'.format(file_dir.split('/')[-2]) not in os.listdir(file_dir + 'picked/'*(not overwrite)):
                    self.write_Exception(file_dir, f, seis_file, exception, mode='w+')
                else:
                    self.write_Exception(file_dir, f, seis_file, exception)
        print('\nSeismograms picked. Bon appetit!')
    
    def cut_Window(self, cross_sec, times, t_i, t_f):
        #init = np.where(times == np.round(t_i, 1))[0][0]
        #end = np.where(times == np.round(t_f, 1))[0][0]
        init = int(np.round(t_i*self.resample_Hz))
        end = int(np.round(t_f*self.resample_Hz))
        
        return cross_sec[init:end]
    
    def shift_Max(self, seis, arrival):
        data = seis.data
        time = seis.times()
        init = np.where(time > (arrival - 1))[0][0]
        end = np.where(time > (arrival + 1))[0][0]
        
        # Interpolate to find "true" maximum
        f = interp1d(time[init:end], data[init:end], kind='cubic')
        t_grid = np.linspace(time[init], time[end-1], num=200)
        amp_max = np.argmax(np.abs(f(t_grid)))
        arrival = t_grid[amp_max]
        
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
    
    def cluster_preds(self, predictions, eps=0.05, min_neighbors=2):
        dbscan = DBSCAN(eps, min_samples=min_neighbors)
        dbscan.fit(predictions.reshape(-1,1))
        clusters, counts = np.unique(dbscan.labels_, return_counts=True)
        if -1 in clusters:
            clusters = clusters[1:]
            counts = counts[1:]
        arrivals = np.zeros(len(clusters))
        arrivals_qual = np.zeros(len(clusters))
        for c in clusters:
            arrivals[c] = np.mean(predictions[dbscan.labels_ ==  c])
            arrivals_qual[c] = counts[c]/self.max_preds
        return arrivals, arrivals_qual
    
    def pick_Phase(self, file_dir, seis_file, phase_name, model, store_header='auto', relevant_preds=1):
    
        seis = obspy.read(file_dir+seis_file)
        seis = seis[0].resample(self.resample_Hz)
        times = seis.times()
        
        phases_in_seis = [seis.stats.sac[k].rstrip(' ') for k in seis.stats.sac.keys() if 'kt' in k]
        phases_headers = [k.lstrip('k') for k in seis.stats.sac.keys() if 'kt' in k]
        phase_var = dict(zip(phases_in_seis, phases_headers))[phase_name]
        
        shift = -seis.stats.sac.b
        begin_time = seis.stats.sac[phase_var] - self.time_window#seis.stats.sac.b
        begin_time = np.round(begin_time + shift, decimals=1)
        end_time = seis.stats.sac[phase_var] + 2*self.time_window#seis.stats.sac.e
        end_time = np.round(end_time + shift, decimals=1)
    
        time_i_grid = np.arange(begin_time, end_time - self.time_window, 1/self.resample_Hz)
        time_f_grid = np.arange(begin_time + self.time_window, end_time, 1/self.resample_Hz)
    
        pos_preds = self.scan(seis, times, time_i_grid, time_f_grid, shift, model)
        neg_preds = self.scan(seis, times, time_i_grid, time_f_grid, shift, model, negative=True)
    
        arrivals_pos, arrivals_pos_qual = self.cluster_preds(pos_preds)
        arrivals_neg, arrivals_neg_qual = self.cluster_preds(neg_preds)
        
        highest_pos_ind = np.argsort(arrivals_pos_qual)[-1]
        highest_neg_ind = np.argsort(arrivals_neg_qual)[-1]
        arrival_pos = arrivals_pos[highest_pos_ind]
        arrival_pos_qual = arrivals_pos_qual[highest_pos_ind]
        arrival_neg = arrivals_neg[highest_neg_ind]
        arrival_neg_qual = arrivals_neg_qual[highest_neg_ind]
        
        t_diff = arrival_pos - arrival_neg
        qual_diff = np.abs(arrival_pos_qual - arrival_neg_qual)
        # If they're this close and of similar quality,
        # then the model is picking the side lobe.
        if (np.abs(t_diff) <= self.time_window) and (qual_diff < 0.1):
            if t_diff < 0:
                arrival = arrival_neg
                arrival_qual = arrival_neg_qual
            else:
                arrival = arrival_pos
                arrival_qual = arrival_pos_qual
        else:
            if arrival_pos_qual > arrival_neg_qual:
                arrival = arrival_pos
                arrival_qual = arrival_pos_qual
            else:
                arrival = arrival_neg
                arrival_qual = arrival_neg_qual
        
        if store_header != 'auto':
            phase_var = store_header
        
        arrival = self.shift_Max(seis, arrival)
        seis.stats.sac[phase_var] = arrival - shift
        seis.stats.sac['k'+phase_var] = phase_name+'ap'
        seis.stats.sac['user'+phase_var[-1]] = np.round(arrival_qual*100)
        seis.stats.sac['kuser0'] = 'PickQual'
        if self.overwrite:
            seis.write(file_dir + seis_file.rstrip('.s_fil') + '.sac')
            os.replace(file_dir + seis_file.rstrip('.s_fil') + '.sac',
                       file_dir + seis_file)
        else:
            seis.write(file_dir + 'picked/'+ seis_file.rstrip('.s_fil') + '_auto' + '.sac')
        
        return
    
    def write_Exception(self, file_dir, file_num, seis_file, exception, mode='a'):
        with open(file_dir + 'picked/'*(not self.overwrite) + '{}_bad.log'.format(file_dir.split('/')[-2]), mode) as log:
            print('File {}: {}'.format(file_num+1, seis_file), file=log)
            print('Error: {}'.format(exception), end='\n\n', file=log)
