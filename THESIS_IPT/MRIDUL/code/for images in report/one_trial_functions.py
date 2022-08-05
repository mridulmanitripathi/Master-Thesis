import h5py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
import peakutils
import scipy
from scipy import fft
from scipy import signal
from scipy import integrate
from scipy.fftpack import fft
from scipy.fftpack import fftfreq
from scipy import stats
from scipy.stats import kurtosis, skew
from scipy.signal import find_peaks
from sklearn import preprocessing
from sklearn.svm import OneClassSVM
import warnings
import random
import math
from math import pi
import seaborn as sns
import openpyxl
from openpyxl import Workbook
from openpyxl import load_workbook
import warnings
from time import process_time
from matplotlib import cm
warnings.filterwarnings("ignore")
plt.rcParams['agg.path.chunksize'] = 10000
plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'font.family': 'Arial'})

# sampling frequencies of the sensors used for data acquistion
sampling_vibration = 50000
sampling_acoustic = 1000


def temp_fft_vib(signal,signal_name, segment_length, sample_rate):
    fft_sample_rate = sample_rate / segment_length

    nr_segments = int(signal.shape[0]/segment_length)                                        #Number of FFT segments

    fft_signal = np.empty((nr_segments,int(segment_length/2)), int)


    for i in range(0, nr_segments):                                                          #Fourier transformation for each data sector with 1000 samples -> 1000/50000 = 20 ms
        segment = signal[0+i*segment_length:segment_length+i*segment_length]
        fft_segment = np.fft.fft(segment)
        fft_amplitude = np.abs(fft_segment[0:int(segment_length/2)])                         #Is defined with 0:500 because data_sector contains 1000 samples, after 500 it jumpes to negative area which is a mirrored picture of positive area (for real numbers), check documentation
        freq = np.fft.fftfreq(segment.size, 1/sample_rate)                                      #Is defined with 0:500 (each value = 50 Hz) because data_sector contains 1000 samples, after 500 it jumpes to negative area which is a copy of positive area (for real numbers), check documentation
        freq = freq[0:int(segment_length/2)]                                                 #Cutting negative half of frequency
        fft_signal[i,:] = fft_amplitude 
    
    xticks = list(np.linspace(0, fft_signal.shape[0]-1, 11, endpoint = True))                #Creates ticks in range of 0 to numbers of rows, each row is one fft for 20 ms
    xlabels = tuple(int((1/fft_sample_rate) * i) for i in xticks)                            #Creates labels for the X-Axis, multiplication with 0.02 because every tick is -> number of row * 20 ms
    
    yticks = list(np.linspace(0, freq.shape[0], 11, endpoint = True))                        #Creates ticks in range of 0 to number of elements in freq
    ylabels = tuple(int(i*freq[1]) for i in yticks)  
    
    
    plt.figure(figsize = (16, 9))
    plt.pcolormesh(np.transpose(fft_signal), vmax = 500)                                     #Creates the diagram, transposes array of coeff so that data is oriented in columns
    plt.colorbar()                                                                           #Creates a colorbar on the right for labeling the amplitude

    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.title('Spectrogram_' + signal_name)

    plt.xticks(xticks,xlabels)
    plt.yticks(yticks,ylabels) 
    #plt.savefig(r'M:\THESIS_IPT\REPORT\images\1_6spectro_' + signal_name +'.png',bbox_inches='tight',dpi=1000)
    plt.show() 
    
    print('freqency shape',freq.shape)
    print('amplitude shape',fft_amplitude.shape)
    print('signal shape',fft_signal.shape)
    
    return fft_signal, fft_sample_rate



def temp_fft_AE(signal, sample_rate):    
    dtype_fft = np.dtype((np.int32, {'x':(np.int16, 0), 'y':(np.int16, 2)})) #Defines a new datatype, 32-bit-data is splitted into two arrays with 16-bit-data

    dset_arr = np.array(signal)                                              #Converts the dataset which is a list into a numpy array
    dset_arr_int = np.int_(dset_arr)                                         #Converts the float data of dataset-array into integer
    dset_new = dset_arr_int.view(dtype = dtype_fft)                          #Creates a new view on the array with the defined datatype, splits the data into two subarrays x and y, dset_arr%512 = a + rest, a = highbyte = amplitude, rest = lowbyte = frequency in [index * 5 kHz]

    freq = dset_new['x']                                                     #x is the array with the frequencies from 0-99 * 5 kHz
    coeff = dset_new['y']                                                    #y is the array with the amplitudes

    freq_re = np.copy(freq)
    coeff_re = np.copy(coeff)
    
    start_ind = np.where(freq == 0)                                          #Searches for entries with 0 and returns the index with [row, column], depends on dimension of input array
    start_ind = int(start_ind[0][0])                                         #First index with freq = 0
    end_ind = np.where(freq == 99)                                           #Searches for entries with 99 and returns the index with [row, column], depends on dimension of input array
    end_ind = int(end_ind[0][-1] + 1)                                        #Last index with freq = 99 and adds 1 so that index of last freq = 0 is taken

    coeff = coeff[start_ind:end_ind]
    freq = freq[start_ind:end_ind]

    coeff = coeff[0:int(len(coeff) / 100)*100]                               #Cuts coeff so that its length is an integer when divided by 100

    coeff = coeff.reshape(int(len(coeff) / 100), 100)                        #Reshapes the array coeff into an array with 100x100 shape

    fft_ae = coeff[:,:]                                                      #Cuts the columns of the array if only the range of specific kHz is important for analysis
    
    xticks = list(np.linspace(0, fft_ae.shape[0]-1, 11, endpoint = True))    #Creates ticks in range of 0 to end of dataset
    xlabels = tuple([int(i / sample_rate) for i in xticks])                  #Creates labels for the X-Axis with values from xticks/1000 (because FFT happens 1000 per second)

    yticks = list(np.linspace(0, fft_ae.shape[1], 11, endpoint = True))      #Creates ticks in range of 0-99
    ylabels = tuple([int(i * 5) for i in yticks])                            #Creates labels for the Y-Axis with values from yticks

    plt.figure(figsize = (16, 9))
    plt.pcolormesh(np.transpose(fft_ae), vmax = 500)                                     #Creates the diagram, transposes array of coeff so that data is oriented in columns
    plt.colorbar()                                                           #Creates a colorbar on the right for labeling the amplitude

    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [kHz]')
    plt.title('Spectrogram_' + 'AE_data')

    plt.xticks(xticks,xlabels)
    plt.yticks(yticks,ylabels)

    #plt.savefig(r'M:\THESIS_IPT\REPORT\images\1_6_AEspectro_vmax.png',bbox_inches='tight',dpi=1000)
    
    return fft_ae



# Extract surface data from ascii file

def extract_surface(filename):
    surface = pd.read_csv(filename, 
                      names = ['measure point', 'longitudinal length', 'surface profile'],
                      sep = ';',
                      header = None)

    measure_point = surface[['measure point']].to_numpy()
    profile = surface[['longitudinal length']].to_numpy()
    
    plt.figure(figsize = (16, 4))
    plt.plot(profile)
    plt.xlabel('Data points')
    plt.ylabel('Magnitude [mm]')
    plt.title('Raw data from Surface Profiler')
#     plt.savefig(r'M:\THESIS_IPT\REPORT\images\1_3_3_surface_raw.png',bbox_inches='tight', dpi=1000)
    
    # determining maximum slope
    slope = []
    for i in range(measure_point.shape[0]-5000):
        m = abs(profile[i+5000]-profile[i])/(measure_point[i+5000]-measure_point[i])
        slope.append(m)
        i = i+1000

    del_ind = np.s_[slope.index(max(slope)):measure_point.shape[0]]   
    measure_point = np.delete(measure_point, del_ind)   
    profile = np.delete(profile, del_ind)
    
    
    #length of surface profile cannot be more than 100mm (size of the workpeice)
    # 100mm is equivalent to 200,000 points
    if measure_point.shape[0]>200000:
        measure_point = measure_point[0:200000]
        profile_norm = profile_norm[0:200000]

    
    plt.figure(figsize = (16, 4))
    plt.plot(profile)
    plt.xlabel('Data points')
    plt.ylabel('Magnitude [mm]')
    plt.title('Surface profile data')
#     plt.savefig(r'M:\THESIS_IPT\REPORT\images\1_3_3_surface_preprocessed.png',bbox_inches='tight',dpi=1000)
    
    return measure_point, profile