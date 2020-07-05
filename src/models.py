#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 15:35:41 2018

@author: jorgeagr
"""
import os
from subprocess import call
import numpy as np
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, BatchNormalization, Input, UpSampling1D, Reshape
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import Sequence
from time import time as clock
import obspy
from src.aux_funcs import check_String, read_Config


class ModelType:
        def __init__(self, model_name):
            config = read_Config('models/conf/{}.conf'.format(model_name))
            self.model_type = config['model_type']
            self.model_name = model_name
            self.model_path = 'models/{}/'.format(self.model_name)
            self.batch_size = config['batch_size']
            self.epochs = config['epochs']
            self.model_iters = config['model_iters']
            self.test_split = float(config['test_split'])
            self.debug = config['debug']
            
            self.files_path = list(map(lambda x: x+'/' if (x[-1] != '/') else x, [config['files_path'],]))[0]
            self.sample_rate = config['sample_rate']
            self.th_arrival_var = config['theory_arrival_var']
            self.arrival_var = config['pick_arrival_var']
            self.window_before = config['window_before']
            self.window_after = config['window_after']
            self.number_shift = config['number_shift']
            self.window_shift = config['window_shift']
            try:
                self.npz_path = config['temp_write_path'] + self.model_name + 'npz/'
            except:
                self.npz_path = self.model_path + 'npz/'
            
            self.total_points = (self.window_before + self.window_after) * self.sample_rate
            
            if self.model_name not in os.listdir('models/'):
                for directory in [self.model_path, self.model_path+'train_logs/']:
                    os.mkdir(directory)
            
            self.trained = self._check_Model()
            return
        
        def _train_Test_Split(self, idnum, seed=None):
            npz_files = np.sort(os.listdir(self.npz_path.format(self.model_name)))
            cutoff = int(len(npz_files) * (1-self.test_split))
            np.random.seed(seed)
            np.random.shuffle(npz_files)
            train_npz_list = npz_files[:cutoff]
            test_npz_list = npz_files[cutoff:]
            np.savez(self.model_path+'train_logs/train_test_split{}'.format(idnum),
                     train=train_npz_list, test=test_npz_list)
            
            return train_npz_list, test_npz_list
        
        def _get_Callbacks(self, epochs):
            stopper = EarlyStopping(monitor='val_loss', min_delta=0, #don't want early stop 
                                    patience=epochs, restore_best_weights=True)
            # Include Checkpoint? CSVLogger?
            return [stopper,]
        
        def load_Model(self):
            self.model = load_model(self.model_path + self.model_name + '.h5')
            return
    
        def save_Model(self):
            if not self.trained:
                self.model.save(self.model_path + self.model_name + '.h5')
            return
        
        def _check_Model(self):
            if self.model_name + '.h5' in os.listdir(self.model_path):
                return True
            else:
                return False

class PickingModel(ModelType):
    
    def __init__(self, model_name):
        super().__init__(model_name)
        return
    
    def __create_Train_Data(self):
        '''
        Function that iterates through seismograms in directory, perform preprocessing,
        create a time window around the arrival time and randomly shift it to augment
        the data set. Save augmented data set as npy files for quicker loading in
        the future. Meant for training/testing data.
        '''
        try:
            os.mkdir(self.npz_path)
        except:
            pass
        files = np.sort(os.listdir(self.files_path))
        gen_whitespace = lambda x: ' '*len(x)
        
        for f, file in enumerate(files):
            if file+'npz' in os.listdir(self.npz_path):
                continue
            else:
                file = check_String(file)
                print_string = 'File ' + str(f+1) + ' / ' + str(len(files)) + '...'
                print('\r'+print_string, end=gen_whitespace(print_string))
                try:
                    seismogram = obspy.read(self.files_path + file)
                except:
                    continue
                seismogram = seismogram[0].resample(self.sample_rate)
                # Begging time
                b = seismogram.stats.sac['b']
                # The beginning time may not be 0, so shift all attribtues to be so
                shift = -b
                b = b + shift
                # End time
                e = seismogram.stats.sac['e'] + shift
                # Theoretical onset arrival time + shift
                if self.th_arrival_var == self.arrival_var:
                    th_arrival = seismogram.stats.sac[self.arrival_var] + shift - np.random.rand() * 20
                else:
                    th_arrival = seismogram.stats.sac[self.th_arrival_var] + shift
                # Picked maximum arrival time + shift
                arrival = seismogram.stats.sac[self.arrival_var] + shift
                
                # Theoretical arrival may be something unruly, so assign some random
                # shift from the picked arrival
                if not (b < th_arrival < e):
                    th_arrival = arrival - 20 * np.random.rand()
                
                amp = seismogram.data
                time = seismogram.times()
                # Shifts + 1 because we want a 0th shift + N random ones
                rand_window_shifts = 2*np.random.rand(self.number_shift+1) - 1 # [-1, 1] interval
                abs_sort = np.argsort(np.abs(rand_window_shifts))
                rand_window_shifts = rand_window_shifts[abs_sort]
                rand_window_shifts[0] = 0
                
                seis_windows = np.zeros((self.number_shift+1, self.total_points, 1))
                arrivals = np.zeros((self.number_shift+1, 1))
                cut_time = np.zeros((self.number_shift+1, 1))
                for i, n in enumerate(rand_window_shifts):
                    rand_arrival = th_arrival - n * self.window_shift
                    init = int(np.round((rand_arrival - self.window_before)*self.sample_rate))
                    end = init + self.total_points
                    if not (time[init] < arrival < time[end]):
                        init = int(np.round((arrival - 15 * np.random.rand() - self.window_before)*self.sample_rate))
                        end = init + self.total_points
                    amp_i = amp[init:end]
                    # Normalize by absolute peak, [-1, 1]
                    amp_i = amp_i / np.abs(amp_i).max()
                    seis_windows[i] = amp_i.reshape(self.total_points, 1)
                    arrivals[i] = arrival - time[init]
                    cut_time[i] = time[init]
                
                np.savez(self.npz_path+'{}'.format(file),
                         seis=seis_windows, arrival=arrivals, cut=cut_time)
            
        return
    
    def __load_Data(self, npz_list, single=False, y_only=False):
        if y_only:
            arr_array = np.zeros((len(npz_list)*(self.number_shift+1)**(not single), 1))
            if single:
                for i, file in enumerate(npz_list):
                    npz = np.load(self.npz_path+file)
                    arr_array[i] = npz['arrival'][0]
            else:
                for i, file in enumerate(npz_list):
                    npz = np.load(self.npz_path+file)
                    arr_array[(self.number_shift+1)*i:(self.number_shift+1)*(i+1)] = npz['arrival']
            return arr_array
        else:
            seis_array = np.zeros((len(npz_list)*(self.number_shift+1)**(not single), self.total_points, 1))
            arr_array = np.zeros((len(npz_list)*(self.number_shift+1)**(not single), 1))
            if single:
                for i, file in enumerate(npz_list):
                    npz = np.load(self.npz_path+file)
                    seis_array[i] = npz['seis'][0]
                    arr_array[i] = npz['arrival'][0]
            else:
                for i, file in enumerate(npz_list):
                    npz = np.load(self.npz_path+file)
                    seis_array[(self.number_shift+1)*i:(self.number_shift+1)*(i+1)] = npz['seis']
                    arr_array[(self.number_shift+1)*i:(self.number_shift+1)*(i+1)] = npz['arrival']
            return seis_array, arr_array
    
    def train_Model(self):
        if self.trained:
            return
        
        if self.debug:
            self.epochs=10
            self.model_iters=1
        
        self.__create_Train_Data()
        
        models = []
        #models_train_means = np.zeros(self.model_iters)
        #models_train_stds = np.zeros(self.model_iters)
        models_test_means = np.zeros(self.model_iters)
        models_test_stds = np.zeros(self.model_iters)
        models_test_final_loss = np.zeros(self.model_iters)
        
        models_train_lpe = np.zeros((self.model_iters, self.epochs))
        models_test_lpe = np.zeros((self.model_iters, self.epochs))
        tick = clock()
        for m in range(self.model_iters):        
            print('Training arrival prediction model', m+1)
            model = self.__rossNet()
            
            callbacks = self._get_Callbacks(self.epochs)
            
            train_files, test_files = self._train_Test_Split(m)
            
            '''
            train_x, train_y = self.__load_Data(train_files)
            test_x, test_y = self.__load_Data(test_files, single=True)
            
            train_hist = model.fit(train_x, train_y,
                                   validation_data=(test_x, test_y),
                                   batch_size=self.batch_size,
                                   epochs=self.epochs,
                                   verbose=2,
                                   callbacks=callbacks)
            '''
            train_generator = PickingDataGenerator(self.npz_path, train_files,
                                                   self.total_points, self.number_shift,
                                                   self.batch_size)
            test_generator = PickingDataGenerator(self.npz_path, test_files, self.total_points,
                                                  self.number_shift, self.batch_size, single=True)
            
            train_hist = model.fit(train_generator,
                                    validation_data=test_generator,
                                    callbacks=callbacks,
                                    verbose=2,)
                                    #use_multiprocessing=True,
                                    #workers=6,)
            
            total_epochs = len(train_hist.history['loss'])
            
            '''
            train_pred = model.predict(train_x)
            test_pred = model.predict(test_x)
            test_loss = model.evaluate(test_x, test_y,
                                       batch_size=self.batch_size, verbose=0)
            '''
            test_y = self.__load_Data(test_files, single=True, y_only=True)
            test_pred = model.predict(test_generator)
            test_loss = model.evaluate(test_generator, verbose=0)
            
            #model_train_diff = np.abs(train_y - train_pred)
            model_test_diff = np.abs(test_y - test_pred)
            #model_train_mean = np.mean(model_train_diff)
            #model_train_std = np.std(model_train_diff)
            model_test_mean = np.mean(model_test_diff)
            model_test_std = np.std(model_test_diff)
            
            #print('Train Error:{:.3f} +/- {:.3f}'.format(model_train_mean, model_train_std))
            print('Test Error: {:.3f} +/- {:.3f}'.format(model_test_mean, model_test_std))
            print('Test Loss: {:.3f}'.format(test_loss))
            
            models.append(model)
            #models_train_means[m] += model_train_mean
            #models_train_stds[m] += model_train_std
            models_test_means[m] += model_test_mean
            models_test_stds[m] += model_test_std
            models_test_final_loss[m] += test_loss
            models_train_lpe[m][:total_epochs] = train_hist.history['loss']
            models_test_lpe[m][:total_epochs] = train_hist.history['val_loss']
        
        #best_model = np.argmin(models_means)
        tock = clock()
        train_time = (tock-tick)/3600 # hours
        best_model = np.argmin(models_test_final_loss)

        with open(self.model_path + 'train_logs/{}_log.txt'.format(self.model_name), 'w+') as log:
            print('\nUsing best model: Model {}\n'.format(best_model), file=log)
            print('Best Model Results:', file=log)
            #print('Training Avg Diff: {:.3f}'.format(models_train_means[best_model]), file=log)
            #print('Training Avg Diff Uncertainty: {:.3f}'.format(models_train_stds[best_model]), file=log)
            print('Testing Avg Diff: {:.3f}'.format(models_test_means[best_model]), file=log)
            print('Testing Avg Diff Uncertainty: {:.3f}'.format(models_test_stds[best_model]), file=log)
            print('Test Loss: {:.3f}'.format(models_test_final_loss[best_model]), file=log)
            print('Total Training Time: {:.2f} hrs'.format(train_time), file=log)
            print('\n')
            if self.debug:
                print('\nmodel saved at this point in no debug', file=log)
                return
        self.model = models[best_model]
        np.savez(self.model_path + 'train_logs/{}_train_history'.format(self.model_name),
                loss=models_train_lpe, val_loss=models_test_lpe, best_model=best_model, train_time=train_time)
        call(['rm','-r',self.npz_path])
        return

    def __rossNet(self):
        '''
        Notes
        ------------
        Ref: https://doi.org/10.1029/2017JB015251 
        '''
        model = Sequential()
        model.add(Conv1D(32, 21, activation='relu',))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        
        model.add(Conv1D(64, 15, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        
        model.add(Conv1D(128, 11, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(1, activation='linear'))
        
        model.compile(loss=Huber(),
                      optimizer=Adam())
        
        return model

class CheckingModel(ModelType):
    
    def __init__(self, model_name):
        super().__init__(model_name)        
        config = read_Config('models/conf/{}.conf'.format(model_name))
        
        self.compression_size = config['compression_size']
        return
    
    def __create_Train_Data(self):
        try:
            os.mkdir(self.npz_path)
        except:
            pass
        files = np.sort(os.listdir(self.files_path))
        gen_whitespace = lambda x: ' '*len(x)
        
        for f, file in enumerate(files):
            if file+'npz' in os.listdir(self.npz_path):
                continue
            else:
                file = check_String(file)
                print_string = 'File ' + str(f+1) + ' / ' + str(len(files)) + '...'
                print('\r'+print_string, end=gen_whitespace(print_string))
                try:
                    seismogram = obspy.read(self.files_path + file)
                except:
                    continue
                seismogram = seismogram[0].resample(self.sample_rate)
                # Begging time
                b = seismogram.stats.sac['b']
                # The beginning time may not be 0, so shift all attribtues to be so
                shift = -b
                b = b + shift
                # End time
                e = seismogram.stats.sac['e'] + shift
                # Theoretical onset arrival time + shift
                if self.th_arrival_var == self.arrival_var:
                    th_arrival = seismogram.stats.sac[self.arrival_var] + shift - np.random.rand() * 20
                else:
                    th_arrival = seismogram.stats.sac[self.th_arrival_var] + shift
                # Picked maximum arrival time + shift
                arrival = seismogram.stats.sac[self.arrival_var] + shift
                
                # Theoretical arrival may be something unruly, so assign some random
                # shift from the picked arrival
                if not (b < th_arrival < e):
                    th_arrival = arrival - 20 * np.random.rand()
                
                amp = seismogram.data
                time = seismogram.times()
                # Shifts + 1 because we want a 0th shift + N random ones
                rand_window_shifts = 2*np.random.rand(self.number_shift+1) - 1 # [-1, 1] interval
                abs_sort = np.argsort(np.abs(rand_window_shifts))
                rand_window_shifts = rand_window_shifts[abs_sort]
                rand_window_shifts[0] = 0
                
                seis_windows = np.zeros((self.number_shift+1, self.total_points, 1))
                for i, n in enumerate(rand_window_shifts):
                    rand_arrival = th_arrival - n * self.window_shift
                    init = int(np.round((rand_arrival - self.window_before)*self.sample_rate))
                    end = init + self.total_points
                    if (end-init < self.total_points):
                        init = init - (self.total_points - (end-init))
                    #    init = int(np.round((arrival - 15 * np.random.rand() - self.window_before)*self.sample_rate))
                    #    end = init + self.total_points
                    amp_i = amp[init:end]
                    # Normalize by absolute peak, [-1, 1]
                    amp_i = amp_i / np.abs(amp_i).max()
                    seis_windows[i] = amp_i.reshape(self.total_points, 1)
                
                np.savez(self.npz_path+'{}'.format(file), seis=seis_windows)
            
        return
    
    def __load_Data(self, npz_list, single=False):
        input_array = np.zeros((len(npz_list)*(self.number_shift+1)**(not single), self.total_points, 1))
        output_array = np.zeros((len(npz_list)*(self.number_shift+1)**(not single), self.total_points, 1))
        if single:
            for i, file in enumerate(npz_list):
                npz = np.load(self.npz_path+file)
                input_array[i] = npz['seis'][0]
                output_array[i] = npz['seis'][0]
        else:
            for i, file in enumerate(npz_list):
                npz = np.load(self.npz_path+file)
                input_array[(self.number_shift+1)*i:(self.number_shift+1)*(i+1)] = npz['seis']
                output_array[(self.number_shift+1)*i:(self.number_shift+1)*(i+1)] = npz['seis']
        return input_array, output_array
    
    def train_Model(self):
        if self.trained:
            return
        
        if self.debug:
            self.epochs=10
            self.model_iters=1
        
        self.__create_Train_Data()
        
        models = []
        models_test_final_loss = np.zeros(self.model_iters)
        
        models_train_lpe = np.zeros((self.model_iters, self.epochs))
        models_test_lpe = np.zeros((self.model_iters, self.epochs))
        tick = clock()
        for m in range(self.model_iters):        
            print('Training arrival prediction model', m+1)
            model = self.__rossNetAE(self.compression_size)
            
            callbacks = self._get_Callbacks(self.epochs)
            
            train_files, test_files = self._train_Test_Split(m)
            '''
            train_x, train_y = self.__load_Data(train_files)
            test_x, test_y = self.__load_Data(test_files)
            
            train_hist = model.fit(train_x, train_y,
                                   validation_data=(test_x, test_y),
                                   batch_size=self.batch_size,
                                   epochs=self.epochs,
                                   verbose=2,
                                   callbacks=callbacks)
            '''
            
            train_generator = CheckingDataGenerator(self.npz_path, train_files,
                                                   self.total_points, self.batch_size)
            test_generator = CheckingDataGenerator(self.npz_path, test_files, self.total_points,
                                                  self.batch_size)
            
            train_hist = model.fit(train_generator,
                                    validation_data=test_generator,
                                    callbacks=callbacks,
                                    verbose=2,)
                                    #use_multiprocessing=True,
                                    #workers=6,)
            
            total_epochs = len(train_hist.history['loss'])
            '''
            train_pred = model.predict(train_x)
            test_pred = model.predict(test_x)
            test_loss = model.evaluate(test_x, test_y,
                                       batch_size=self.batch_size, verbose=0)
            '''
            #train_pred = model.predict(train_generator)
            #test_pred = model.predict(test_generator)
            test_loss = model.evaluate(test_generator, verbose=0)
            
            #model_train_diff = np.abs(train_y - train_pred)
            #model_test_diff = np.abs(test_y - test_pred)
            #model_train_mean = np.mean(model_train_diff)
            #model_train_std = np.std(model_train_diff)
            #model_test_mean = np.mean(model_test_diff)
            #model_test_std = np.std(model_test_diff)
            
            #print('Train Error: {:.3f} +/- {:.3f}'.format(model_train_mean, model_train_std))
            #print('Test Error: {:.3f} +/- {:.3f}'.format(model_test_mean, model_test_std))
            print('Test Loss: {:.3f}'.format(test_loss))
            
            models.append(model)
            #models_train_means[m] += model_train_mean
            #models_train_stds[m] += model_train_std
            #models_test_means[m] += model_test_mean
            #models_test_stds[m] += model_test_std
            models_test_final_loss[m] += test_loss
            models_train_lpe[m][:total_epochs] = train_hist.history['loss']
            models_test_lpe[m][:total_epochs] = train_hist.history['val_loss']
        
        #best_model = np.argmin(models_means)
        tock = clock()
        train_time = (tock-tick)/3600 # hours
        best_model = np.argmin(models_test_final_loss)

        with open(self.model_path + 'train_logs/{}_log.txt'.format(self.model_name), 'w+') as log:
            print('\nUsing best model: Model {}\n'.format(best_model), file=log)
            print('Best Model Results:', file=log)
            #print('Training Avg Diff: {:.3f}'.format(models_train_means[best_model]), file=log)
            #print('Training Avg Diff Uncertainty: {:.3f}'.format(models_train_stds[best_model]), file=log)
            #print('Testing Avg Diff: {:.3f}'.format(models_test_means[best_model]), file=log)
            #print('Testing Avg Diff Uncertainty: {:.3f}'.format(models_test_stds[best_model]), file=log)
            print('Test Loss: {:.3f}'.format(models_test_final_loss[best_model]), file=log)
            print('Total Training Time: {:.2f} hrs'.format(train_time), file=log)
            print('\n')
            if self.debug:
                print('\nmodel saved at this point in no debug', file=log)
                return
        self.model = models[best_model]
        np.savez(self.model_path + 'train_logs/{}_train_history'.format(self.model_name),
                loss=models_train_lpe, val_loss=models_test_lpe, best_model=best_model, train_time=train_time)
        
        call(['rm','-r',self.npz_path])
        return

    def __rossNetAE(self, compression_size):
        '''
        Notes
        ------------
        Main architecture idea:
        Ref: https://doi.org/10.1029/2017JB015251
        '''
        input_seis = Input(shape=(self.total_points, 1))

        conv1 = Conv1D(32, kernel_size=21, strides=1,
                         activation='relu', padding='same')(input_seis)
        bn1 = BatchNormalization()(conv1)
        max1 = MaxPooling1D(pool_size=2)(bn1)
    
        conv2 = Conv1D(64, kernel_size=15, strides=1,
                         activation='relu', padding='same')(max1)
        bn2 = BatchNormalization()(conv2)
        max2 = MaxPooling1D(pool_size=2)(bn2)
    
        conv3 = Conv1D(128, kernel_size=11, strides=1,
                         activation='relu', padding='same')(max2)
        bn3 = BatchNormalization()(conv3)
        max3 = MaxPooling1D(pool_size=2)(bn3)
    
        flattened = Flatten()(max3)
        
        encoding = Dense(compression_size, activation='sigmoid')(flattened)
        
        expanded = Dense(max3.shape.as_list()[1] * max3.shape.as_list()[2], activation='relu')(encoding)
        
        reshaped = Reshape(max3.shape.as_list()[1:])(expanded)
        
        up1 = UpSampling1D(size=2)(reshaped)
        bn_up1 = BatchNormalization()(up1)
        conv_up1 = Conv1D(128, kernel_size=11, strides=1,
                         activation='relu', padding='same')(bn_up1)
    
        up2 = UpSampling1D(size=2)(conv_up1)
        bn_up2 = BatchNormalization()(up2)
        conv_up2 = Conv1D(64, kernel_size=15, strides=1,
                          activation='relu', padding='same')(bn_up2)
    
        up3 = UpSampling1D(size=2)(conv_up2)
        bn_up3 = BatchNormalization()(up3)
        conv_up3 = Conv1D(32, kernel_size=21, strides=1,
                          activation='relu', padding='same')(bn_up3)
        # sigmoid? or tanh? or maybe something else
        decoding = Conv1D(1, kernel_size=21, strides=1,
                          activation='linear', padding='same')(conv_up3)
        
        model = Model(input_seis, decoding)
        
        model.compile(loss='mean_absolute_error',
                  optimizer=Adam(1e-4))
        
        return model
    
class CheckingDataGenerator(Sequence):
    '''
    Based on an implementation by Shervine Amidi
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    '''
    def __init__(self, npy_path, list_IDs, seismo_size, batch_size=128, n_channels=1, shuffle=True):
        'Initialization'
        self.path = npy_path
        self.dim = (1, seismo_size, 1)
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.seismo_size = seismo_size

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(list_IDs_temp)

        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        x = np.zeros((self.batch_size, self.seismo_size, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            x[i,] = np.load(self.path + ID)['seis']

        return x, x
    
class PickingDataGenerator(Sequence):
    '''
    Based on an implementation by Shervine Amidi
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    '''
    def __init__(self, npy_path, list_IDs, seismo_size, number_shifts, batch_size, n_channels=1, single=False, shuffle=True):
        'Initialization'
        self.path = npy_path
        self.single = single
        #self.dim = (1, seismo_size, 1)
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.seismo_size = seismo_size
        self.number_shifts = number_shifts
        if not self.single:
            self.list_IDs = self.gen_Variations()
        self.on_epoch_end()

    def gen_Variations(self):
        list_IDs_temp = []
        for i in range(len(self.list_IDs)):
            for j in range(self.number_shifts+1):
                list_IDs_temp.append(self.list_IDs[i]+str(j))
        return list_IDs_temp

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(list_IDs_temp)

        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        x = np.zeros((self.batch_size, self.seismo_size, self.n_channels))
        y = np.zeros((self.batch_size, 1))

        # Generate data
        if not self.single:
            for i, ID in enumerate(list_IDs_temp):
                x[i,] = np.load(self.path + ID[:-1])['seis'][int(ID[-1])]
                y[i,] = np.load(self.path + ID[:-1])['arrival'][int(ID[-1])]
        else:
            for i, ID in enumerate(list_IDs_temp):
                x[i,] = np.load(self.path + ID)['seis'][0]
                y[i,] = np.load(self.path + ID)['arrival'][0]

        return x, y