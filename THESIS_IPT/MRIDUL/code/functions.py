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


# computing Fourier Transformations of the vibration data

# Approach 1
def fft_signal(array, sampling):
    Xf_mag = np.abs(np.fft.fft(array))
    freqs = np.fft.fftfreq(len(Xf_mag), d=1.0/sampling)

    #taking half of the data (same in negative)
    bound = Xf_mag.shape[0]//2      
    Xf_mag_temp = Xf_mag[0:bound]
    freqs_temp = freqs[0:bound]
    
    return freqs_temp, Xf_mag_temp


# Approach 2
def fftVibration(signal, segment_length, sample_rate):
    freq = np.zeros(500)
    fft_amplitude = np.zeros(500)
    fft_sample_rate = sample_rate / segment_length
    nr_segments = int(signal.shape[0]/segment_length)                                        #Number of FFT segments
    fft_signal = np.empty((nr_segments,int(segment_length/2)), int)

    for i in range(0, nr_segments):                                                          #Fourier transformation for each data sector with 1000 samples -> 1000/50000 = 20 ms
        segment = signal[0+i*segment_length:segment_length+i*segment_length]
        fft_segment = np.fft.fft(segment)
        fft_amplitude = np.abs(fft_segment[0:int(segment_length/2)])                         #Is defined with 0:500 because data_sector contains 1000 samples, after 500 it jumpes to negative area which is a mirrored picture of positive area (for real numbers), check documentation
        freq = np.fft.fftfreq(segment.size, 1/sample_rate)                                   #Is defined with 0:500 (each value = 50 Hz) because data_sector contains 1000 samples, after 500 it jumpes to negative area which is a copy of positive area (for real numbers), check documentation
        freq = freq[0:int(segment_length/2)]                                                 #Cutting negative half of frequency
        fft_signal[i,:] = fft_amplitude 
    
    return freq, fft_amplitude


# Approach 3: Computing Short Time Fourier Transform
def stft_vibration_signal(signal, window_length_time, sample_rate):
    window_length_samples = sample_rate*window_length_time                    # determining number of data points in the window
    stft_per_second = sample_rate / window_length_samples                     # computing the sampling rate of the STFT
    
    f, t, Zxx = scipy.signal.stft(x=signal, fs=sample_rate, nperseg=window_length_samples, noverlap=0, window='boxcar')
    stft_magnitude = np.abs(Zxx) 
  
    return f, t, stft_magnitude, stft_per_second 



# computing Short Time Fourier Transformations of the acoustic data

def fftAE(signal, sample_rate):    
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

    # high pass filter
#    fft_ae[:,0] = 0                                                         # Filtering out frequencies in the range 0 kHz - 5 kHz
    
    return fft_ae


# Butterworth band pass filter

def butterworth_bandpass(array, sampling, low_freq, high_freq):
    # computing nyquist rate 
    nyq = 0.5*sampling                                                                   
    low = low_freq/nyq
    high = high_freq/nyq
    order = 5
    # stores the data in the frequency range of lower frequency to high frequency
    sos = signal.butter(order, [low,high], analog=False, btype='band', output='sos')      
    # after keeping the probable chatter region
    filtered_array = signal.sosfilt(sos,array)                                            
    
    return filtered_array


# extracting milling data for each experimental trial

def extract_mill(array, int_size, cutoff):
    # computing the number of windows that can be generated with given interval size (int_size)
    quo = array.shape[0]//int_size
    new_range = quo*int_size
    # genrating uniform intervals for the vibration data 
    interval = np.arange(0, new_range, int_size)
    k = 0
    peak_data = []
    
    # storing peak values in a interval that are greater than the cutoff value
    for i in interval:
        peak=[]
        if (array[i:interval[k+1]-1].max()<cutoff):
            for j in range(i,interval[k+1]-1):
                if (array[j]<cutoff):
                    peak.append(j)
                
            if (len(peak)in range(10000)):
                peak_data.append(j)
    
        k = k+1  
        if (i==interval[-2]):
            break
    
    # determining indices of start and end points 
    index_trial_start = []
    index_trial_end = []
    
    for x, y in zip(peak_data, peak_data[1:]):
        if abs(x-y)>10000:
            index_trial_start.append(x)
            index_trial_end.append(y)
            
    index_trial_start = np.asarray(index_trial_start)
    index_trial_end = np.asarray(index_trial_end)
    
    
    # removing undesired start and end points
    del_outlier = (index_trial_end - index_trial_start).max()
    index_trial_start_new = np.copy(index_trial_start)
    index_trial_end_new = np.copy(index_trial_end)
    
    for i in range(len(index_trial_start)-1):
        if ((abs(index_trial_end[i]-index_trial_start[i])<(del_outlier-int_size))):
            index_trial_start_new = np.delete(index_trial_start, i)
            index_trial_end_new = np.delete(index_trial_end, i)
    
    
    return index_trial_start_new, index_trial_end_new



# Exctracting indices of start and end of milling operation

def milling_ind(array, int_size, cutoff):
    ind_start, ind_end = extract_mill(array, int_size, cutoff)
    inc_cutoff = 1
    
    # since each milled surface has 5 lines of milling, there are 5 start points and 5 end points in the vibration data
    for i in range(15):
        if ind_start.shape[0]<5:
            inc_cutoff = inc_cutoff+1
            ind_start, ind_end = extract_mill(array, int_size = 10000, cutoff = inc_cutoff)
        if ind_start.shape[0]==5:
            break   
    return  ind_start, ind_end



# remove air cut data
def remove_aircut(array, int_size, cutoff):
    vibration_x_no_aircut = []
    quo = array.shape[0]//int_size
    new_range = quo*int_size
    interval = np.arange(0, new_range, int_size)
    k = 0
    
    for i in interval:
        peak = []
        if (abs(array[i:interval[k+1]-1]).max()>cutoff):
            for j in range(i,interval[k+1]):
                if (abs(array[j])>cutoff):
                    peak.append(j)
                
            if len(peak)>2:
                for j in range(i,interval[k+1]):
                    vibration_x_no_aircut.append(array[j])        
        
        k = k+1  
        if (i==interval[-2]):
            break
            
    return vibration_x_no_aircut


# remove milling data
def remove_milling(array, int_size, cutoff):
    vibration_x_no_milling = []
    quo = array.shape[0]//int_size
    new_range = quo*int_size
    interval = np.arange(0, new_range, int_size)
    k = 0
    
    for i in interval:
        peak = []
        if (abs(array[i:interval[k+1]-1]).max()<cutoff):
            for j in range(i,interval[k+1]):
                if (abs(array[j])<cutoff):
                    peak.append(j)
                
            if len(peak)>2:
                for j in range(i,interval[k+1]):
                    vibration_x_no_milling.append(array[j])        
        
        k = k+1  
        if (i==interval[-2]):
            break
            
    return vibration_x_no_milling


# extracting milling data
def milling_data(array, ind_start, ind_end):
    mill_data = []
    for i in range(len(ind_start)):
        temp = array[ind_start[i]:ind_end[i]]
        mill_data.extend(temp)
        
    return mill_data


# extracting aircut data
def aircut_data(array, ind_start, ind_end):
    air_data = []
    for i in range(len(ind_start)):
        if i!=4:
            temp = array[ind_end[i]:ind_start[i+1]]
            air_data.extend(temp)
        else:
            break
    
    # Too much of aircut data would mean that ML model wont learn to differentiate effectively
    
    sum_mill = 0
    for i in range(len(ind_start)):
        sum_mill = sum_mill + (ind_end[i]-ind_start[i])
    
    if (sum_mill>len(air_data)):
        diff = sum_mill-len(air_data)
        
        temp2 = array[ind_end[4]:ind_end[4]+(diff//2)]
        air_data.extend(array[0:(diff//2)])
        air_data.extend(temp2)
        
    return air_data



# generating labels for milling data and aircut data

# Approach 1
def aircut_milling_separation(array, ind_start, ind_end):
    air_data = pd.DataFrame(columns=['vibration_x','vibration_y','vibration_z', 'aircut0'])
    mill_data = pd.DataFrame(columns=['vibration_x','vibration_y','vibration_z', 'aircut0'])
    
    md_x = milling_data(array[0], ind_start,ind_end)
    md_y = milling_data(array[1], ind_start,ind_end)
    md_z = milling_data(array[2], ind_start,ind_end)
    
    ad_x = aircut_data(array[0], ind_start,ind_end)
    ad_y = aircut_data(array[1], ind_start,ind_end)
    ad_z = aircut_data(array[2], ind_start,ind_end)
    
    mill_data['vibration_x'] = md_x
    mill_data['vibration_y'] = md_y
    mill_data['vibration_z'] = md_z
    # labelling milling data as 1
    mill_data['aircut0'] = 0
    
    air_data['vibration_x'] = ad_x
    air_data['vibration_y'] = ad_y
    air_data['vibration_z'] = ad_z
    # labelling aircut data as 0
    air_data['aircut0'] = 1
   
    return air_data, mill_data


# Approach 2
def air_mill_classify(array, ind_start, ind_end):
    air_mill_data = pd.DataFrame(columns=['vibration_x','vibration_y','vibration_z', 'aircut0'])
    
    air_mill_data['vibration_x'] = array[0]
    air_mill_data['vibration_y'] = array[1]
    air_mill_data['vibration_z'] = array[2]
    # labelling aircut data as 0
    air_mill_data['aircut0'] = 0
    
    # labelling milling data as 1
    for i in range(len(ind_start)):
        air_mill_data.loc[air_mill_data.index[ind_start[i]:ind_end[i]], 'aircut0'] = 1
            
    return air_mill_data



# Extract surface data from ascii file generated by the surface roughness experiments

def extract_surface(filename):
    surface = pd.read_csv(filename, 
                      names = ['measure point', 'longitudinal length', 'surface profile'],
                      sep = ';',
                      header = None)

    # indices of the measured points
    measure_point = surface[['measure point']].to_numpy()
    # surface roughness values
    profile = surface[['longitudinal length']].to_numpy()
    
#     plt.figure(figsize = (16, 4))
#     plt.plot(profile)
#     plt.xlabel('Data points')
#     plt.ylabel('Magnitude [mm]')
#     plt.title('Raw data from Surface Profiler')
#     plt.savefig(r'M:\THESIS_IPT\REPORT\images\1_3_3_surface_raw.png',bbox_inches='tight', dpi=1000)
    
    # determining maximum slope
    slope = []
    for i in range(measure_point.shape[0]-5000):
        m = abs(profile[i+5000]-profile[i])/(measure_point[i+5000]-measure_point[i])
        slope.append(m)
        i = i+1000

    # deleting the data points after the maximum slope (undesired data points)
    del_ind = np.s_[slope.index(max(slope)):measure_point.shape[0]]   
    measure_point = np.delete(measure_point, del_ind)   
    profile = np.delete(profile, del_ind)
    
    
    #length of surface profile cannot be more than 100mm (size of the workpeice)
    # 100mm is equivalent to 200,000 points
    if measure_point.shape[0]>200000:
        measure_point = measure_point[0:200000]
        profile_norm = profile_norm[0:200000]
    
#     plt.figure(figsize = (16, 1))
#     plt.plot(profile_norm)
    
#     plt.figure(figsize = (16, 4))
#     plt.plot(profile)
#     plt.xlabel('Data points')
#     plt.ylabel('Magnitude [mm]')
#     plt.title('Surface profile data')
#     plt.savefig(r'M:\THESIS_IPT\REPORT\images\1_3_3_surface_preprocessed.png',bbox_inches='tight',dpi=1000)
    
    return measure_point, profile



# To get the baseline of the measured surface profile

def generate_baseline(win, surface):
    baseline_value=[]
    num_win = surface.shape[0]//win
    
    # generating the baseline
    for i in range(num_win):
        temp_surf = surface[i*win:(win+i*win)]
        temp_base = peakutils.baseline(temp_surf)
        baseline_value.append(temp_base)
    
    # generating baseline for surface data left by the last window
    if surface.shape[0]>=(win*num_win):
        temp_surf = surface[num_win*win:]
        temp_base = peakutils.baseline(temp_surf)
        baseline_value.append(temp_base)
        
    baseline_values = np.concatenate(baseline_value, axis=None)
        
    return baseline_values  



# Saving the image of surface profile and the baseline

def save_surface(file_loc, file, location):
    filename_surface = file_loc +'//'+ file
    measure_points, profile = extract_surface(filename_surface)
    baseline = generate_baseline(win=9999, surface=profile)
    
    fig=plt.figure()
    plt.figure(figsize = (16, 9))
    plt.plot(profile)
    plt.plot(baseline, '--r')
    plt.legend(['surface profile', 'Baseline'])
    plt.close(fig)
    name = os.path.splitext(file)[0]
    #plt.savefig(location + '//' + name)
    
    
    
    
# for computing the mean peak value of the true surface profile (whole)

def save_surface_peaks_mean(file_loc, file):
    filename_surface = file_loc +'//'+ file
    # extracting surface measurements
    measure_points, profile = extract_surface(filename_surface)
    # computing baseline
    baseline = generate_baseline(win=5000, surface=profile)
    # determining true surface profile
    diff = profile - baseline
    # computing peaks of the true surface profile
    peaks,_ = find_peaks(diff, height=0.0001)
    # computing mean of the peak values
    peak_mean = diff[peaks].mean()
    print(peak_mean)
    
    return peak_mean


# Merging vibration data with the surface profile data

def merge_vibration_surface(array, surf_combined, ind_start, ind_end):
    merged_data = pd.DataFrame(columns=['vibration_x','vibration_y','vibration_z', 'surface_roughness'])
    
    merged_data['vibration_x'] = array[0]
    merged_data['vibration_y'] = array[1]
    merged_data['vibration_z'] = array[2]
    merged_data['surface_roughness'] = 0
    
    for i in range(len(ind_start)):
        num_data = ind_end[i]-ind_start[i]
        # resampling the surface data to match the number of points in the milling part of the vibration data
        resampled_surface = signal.resample(surf_combined[i], num_data)
        merged_data.loc[merged_data.index[ind_start[i]:ind_end[i]], 'surface_roughness'] = resampled_surface
        
    return merged_data



# Computing time domain features

# RMS
def rms_time(array):
    array_rms = np.sqrt(np.mean(array**2))
    return array_rms


# Kurtosis
def kurt_time(array):
    array_kurt = np.sum((array-array.mean())**4)/(array.std()**4)
    return array_kurt


# Skewness
def skew_time(array):
    array_skew = np.sum((array-array.mean())**3)/(array.std()**3)
    return array_skew


# Mean
def mean_time(array):
    array_mean = array.mean()
    return array_mean


# Standard deviation
def std_time(array):
    array_std = array.std()
    return array_std


# Peak
def peak_time(array):
    array_peak = np.abs(array).max()
    return array_peak


# Crest factor
def crest_time(array):
    peak = peak_time(array)
    rms = rms_time(array)
    array_crest = peak/rms
    return array_crest


# Clearance factor
def clearance_time(array):
    peak = peak_time(array)
    den = (np.sqrt(np.abs(array).mean()))**2
    array_clear = peak/den
    return array_clear


# Shape factor
def shape_time(array):
    rms = rms_time(array)
    den = np.abs(array).mean()
    array_shape = rms/den
    return array_shape


# Impulse factor
def impulse_time(array):
    peak = peak_time(array)
    den = np.abs(array).mean()
    array_impulse = peak/den
    return array_impulse

# computing fractal dimension using SFM
def frac_dimen(array, sampling_vibration):
    N = array.shape[0]
    n = np.arange(1,1000,400)
    
    S=np.zeros(3)
    tau = (n/sampling_vibration)
    diff = 0
    m = 0
    
    for j in n:
        for i in range(N-j):
            diff = diff + (array[i+j]-array[i])**2
        
        temp = (1/(N-j))*diff
        S[m] = temp
        m = m + 1
    
    logS = np.log(S)
    logTau = np.log(tau)  
    slope, intercept = np.polyfit(logTau,logS,1)
    FracDim = 2-(slope/2)
    
    return FracDim

# Square root mean
def sqrtmean_time(array):
    array_sqrtmean = ((np.sqrt(np.abs(array))).mean())**2
    return array_sqrtmean

# SRM shape factor
def srmshape_time(array):
    sqrtmean = sqrtmean_time(array)
    den = np.abs(array).mean()
    array_srmshape = sqrtmean/den
    return array_srmshape

# Latitude factor
def latitude_time(array):
    peak = peak_time(array)
    sqrtmean = sqrtmean_time(array)
    array_latitude = peak/sqrtmean
    return array_latitude

# Fifth moment
def fifth_time(array):
    array_fifth = np.sum((array-array.mean())**5)/(array.std()**5)
    return array_fifth

# Sixth moment
def sixth_time(array):
    array_sixth = np.sum((array-array.mean())**6)/(array.std()**6)
    return array_sixth




# Computing freqency domain features using power spectrum 

# Mean square frequency
def msf_freq(f,m):
    num = np.sum((f**2)*(m**2))
    den = np.sum((m**2))
    
    msf = num/den
    return msf


# One step autocorrelation
def osac_freq(f,m):
    delta_t = (1/(f[2])) - (1/(f[1]))
    func1 = np.vectorize(math.radians)
    rad = func1(2*pi*f*delta_t)
    func2 = np.vectorize(math.cos)
    num = np.sum(func2(rad)*(m**2))
    den = np.sum((m**2))
    
    osac = num/den
    return osac


# Frequency centre
def fc_freq(f,m):
    num = np.sum(f*(m**2))
    den = np.sum((m**2))
    
    fc = num/den
    return fc


# Standard frequency
def sf_freq(f,m):
    fc = fc_freq(f,m)
    num = np.sum(((f-fc)**2)*(m**2))
    den = np.sum((m**2))
    
    sf = num/den
    return sf

# Magnitude of frequencies as feature
def freq_as_feature(f,m):
    FreqFeat = np.zeros(5)
    for i in range(5):
        temp = m[i*100:i*100 + 100]
        FreqFeat[i] = (temp**2).mean()
    return FreqFeat




# input of 1000 data points (20 ms) is given
def add_features(dataframe, sampling_vibration, win, segment_length):
    features = []
    vibration_x = np.asarray(dataframe['vibration_x'])
    vibration_y = np.asarray(dataframe['vibration_y'])
    vibration_z = np.asarray(dataframe['vibration_z'])
    dataframe[dataframe['surface_roughness']<=0]=0.001  # to avoid division by 0
    
    rms_x = rms_time(vibration_x)
    rms_y = rms_time(vibration_y)
    rms_z = rms_time(vibration_z)
    
    kurt_x = kurt_time(vibration_x)
    kurt_y = kurt_time(vibration_y)
    kurt_z = kurt_time(vibration_z)
    
    skew_x = skew_time(vibration_x)
    skew_y = skew_time(vibration_y)
    skew_z = skew_time(vibration_z)
    
    mean_x = mean_time(vibration_x)
    mean_y = mean_time(vibration_y)
    mean_z = mean_time(vibration_z)
    
    std_x = std_time(vibration_x)
    std_y = std_time(vibration_y)
    std_z = std_time(vibration_z)
    
    peak_x = peak_time(vibration_x)
    peak_y = peak_time(vibration_y)
    peak_z = peak_time(vibration_z)
    
    crest_x = crest_time(vibration_x)
    crest_y = crest_time(vibration_y)
    crest_z = crest_time(vibration_z)
    
    clear_x = clearance_time(vibration_x)
    clear_y = clearance_time(vibration_y)
    clear_z = clearance_time(vibration_z)
    
    shape_x = shape_time(vibration_x)
    shape_y = shape_time(vibration_y)
    shape_z = shape_time(vibration_z)
    
    impulse_x = impulse_time(vibration_x)
    impulse_y = impulse_time(vibration_y)
    impulse_z = impulse_time(vibration_z)
    
    # Approach 1 for computing the fourier transformation of the vibration data
#     fx, mx = fft_signal(vibration_x, sampling_vibration)
#     fy, my = fft_signal(vibration_y, sampling_vibration)
#     fz, mz = fft_signal(vibration_z, sampling_vibration)

    # Approach 2 for computing the fourier transformation of the vibration data
#     fx, mx = fftVibration(vibration_x, segment_length, sampling_vibration)
#     fy, my = fftVibration(vibration_y, segment_length, sampling_vibration)
#     fz, mz = fftVibration(vibration_z, segment_length, sampling_vibration)
    
    # Approach 3 for computing the fourier transformation of the vibration data
    fx,_,m_x,_= stft_vibration_signal(signal=vibration_x, window_length_time = segment_length/sampling_vibration, sample_rate=sampling_vibration)
    mx = m_x[:,1]
    fy,_,m_y,_= stft_vibration_signal(signal=vibration_y, window_length_time = segment_length/sampling_vibration, sample_rate=sampling_vibration)
    my = m_y[:,1]
    fz,_,m_z,_= stft_vibration_signal(signal=vibration_z, window_length_time = segment_length/sampling_vibration, sample_rate=sampling_vibration)
    mz = m_z[:,1]
    

    msf_x = msf_freq(fx, mx)
    msf_y = msf_freq(fy, my)
    msf_z = msf_freq(fz, mz)
    
    osac_x = osac_freq(fx, mx)
    osac_y = osac_freq(fy, my)
    osac_z = osac_freq(fz, mz)
    
    fc_x = fc_freq(fx, mx)
    fc_y = fc_freq(fy, my)
    fc_z = fc_freq(fz, mz)
    
    sf_x = sf_freq(fx, mx)
    sf_y = sf_freq(fy, my)
    sf_z = sf_freq(fz, mz)
    
    
    # computing baseline
    baseline = generate_baseline(win, surface=dataframe['surface_roughness'])
    # determining true surface profile
    diff = dataframe['surface_roughness'] - baseline
    diff = pd.Series.to_numpy(diff)
    # computing peaks of the surface profile
    peaks,_ = find_peaks(diff, height=0.0001)
    # computing mean of the peak values
    avg_peak = diff[peaks].mean()
    
    frac_dim_x = frac_dimen(vibration_x, sampling_vibration)
    frac_dim_y = frac_dimen(vibration_y, sampling_vibration)
    frac_dim_z = frac_dimen(vibration_z, sampling_vibration)
    
    fifth_x = fifth_time(vibration_x)
    fifth_y = fifth_time(vibration_y)
    fifth_z = fifth_time(vibration_z)
    
    sixth_x = sixth_time(vibration_x)
    sixth_y = sixth_time(vibration_y)
    sixth_z = sixth_time(vibration_z)
    
    freqfeat_x = freq_as_feature(fx, mx)
    freqfeat_y = freq_as_feature(fy, my)
    freqfeat_z = freq_as_feature(fz, mz)
    
    # arranging all features (time domain + frequency domain) of vibration data
    features=[[rms_x, rms_y, rms_z, 
              kurt_x, kurt_y, kurt_z,
              skew_x, skew_y, skew_z,
              mean_x, mean_y, mean_z,
              std_x, std_y, std_z,
              peak_x, peak_y, peak_z,
              crest_x, crest_y, crest_z,
              clear_x, clear_y, clear_z,
              shape_x, shape_y, shape_z,
              impulse_x, impulse_y, impulse_z,
              msf_x, msf_y, msf_z,
              osac_x, osac_y, osac_z,
              fc_x, fc_y, fc_z,
              sf_x, sf_y, sf_z,
              avg_peak, 
              frac_dim_x, frac_dim_y, frac_dim_z,
              fifth_x, fifth_y, fifth_z,
              sixth_x, sixth_y, sixth_z,
              freqfeat_x[0], freqfeat_x[1], freqfeat_x[2], freqfeat_x[3], freqfeat_x[4],
              freqfeat_y[0], freqfeat_y[1], freqfeat_y[2], freqfeat_y[3], freqfeat_y[4],
              freqfeat_z[0], freqfeat_z[1], freqfeat_z[2], freqfeat_z[3], freqfeat_z[4]]]
    
    return features 



# Computing AE sensor data freqency domain features using power spectrum 

# Mean square frequency
def msf_AEfreq(f,m):
    num = np.sum((f**2)*(m**2))
    den = np.sum((m**2))
    
    msf = num/den
    return msf


# One step autocorrelation
def osac_AEfreq(f,m):
    delta_t = 1/(f[2]-f[1])
    func1 = np.vectorize(math.radians)
    rad = func1(2*pi*f*delta_t)
    func2 = np.vectorize(math.cos)
    num = np.sum(func2(rad)*(m**2))
    den = np.sum((m**2))
    
    osac = num/den
    return osac


# Frequency centre
def fc_AEfreq(f,m):
    num = np.sum(f*(m**2))
    den = np.sum((m**2))
    
    fc = num/den
    return fc


# Standard frequency
def sf_AEfreq(f,m):
    fc = fc_AEfreq(f,m)
    num = np.sum(((f-fc)**2)*(m**2))
    den = np.sum((m**2))
    
    sf = num/den
    return sf

# Magnitude of frequencies as feature
def freq_as_featureAE(f,m):
    FreqFeatAE = np.zeros(5)
    for i in range(5):
        temp = m[i*20:i*20 + 20]
        FreqFeatAE[i] = (temp**2).mean()
    return FreqFeatAE




# input of 2000 data points of AE sensor is given
def add_AEfeatures(f,m):
    features = []

    msf_AE = msf_AEfreq(f,m)    
    osac_AE = osac_AEfreq(f,m)
    fc_AE = fc_AEfreq(f,m)  
    sf_AE = sf_AEfreq(f,m)
    freqfeat = freq_as_featureAE(f,m)
    
    # arranging all features (only frequency domain) of acoustic data
    features=[[msf_AE, osac_AE, fc_AE, sf_AE, freqfeat[0], freqfeat[1], freqfeat[2], freqfeat[3], freqfeat[4]]]
    
    return features 


# transforming into classification data
def class_transform(class_num, data):
    # target variable
    a = data.columns.get_loc("avg_peak")
    
    # for 10-class classification
    if class_num==10:
        for i in range(data.shape[0]):
            if data.iloc[i,a]==0.0001:
                data.iloc[i,a] = 0
    
            elif data.iloc[i,a]>0.0001 and data.iloc[i,a]<0.001:
                data.iloc[i,a] = 1
        
            elif data.iloc[i,a]>=0.001 and data.iloc[i,a]<0.002:
                data.iloc[i,a] = 2
        
            elif data.iloc[i,a]>=0.002 and data.iloc[i,a]<0.003:
                data.iloc[i,a] = 3
        
            elif data.iloc[i,a]>=0.003 and data.iloc[i,a]<0.004:
                data.iloc[i,a] = 4
        
            elif data.iloc[i,a]>=0.004 and data.iloc[i,a]<0.005:
                data.iloc[i,a] = 5
        
            elif data.iloc[i,a]>=0.005 and data.iloc[i,a]<0.006:
                data.iloc[i,a] = 6
        
            elif data.iloc[i,a]>=0.006 and data.iloc[i,a]<0.007:
                data.iloc[i,a] = 7
        
            elif data.iloc[i,a]>=0.007 and data.iloc[i,a]<0.008:
                data.iloc[i,a] = 8
        
            elif data.iloc[i,a]>=0.008 and data.iloc[i,a]<0.009:
                data.iloc[i,a] = 9
        
            elif data.iloc[i,a]>=0.009:
                data.iloc[i,a] = 10
    
    # for 5-class classification
    if class_num==5:
        for i in range(data.shape[0]):
            if data.iloc[i,a]==0.0001:
                data.iloc[i,a] = 0
    
            elif data.iloc[i,a]>0.0001 and data.iloc[i,a]<0.001:
                data.iloc[i,a] = 1
    
            elif data.iloc[i,a]>0.001 and data.iloc[i,a]<0.0025:
                data.iloc[i,a] = 2
            
            elif data.iloc[i,a]>=0.0025 and data.iloc[i,a]<0.0031:
                data.iloc[i,a] = 3
        
            elif data.iloc[i,a]>=0.0031 and data.iloc[i,a]<0.004:
                data.iloc[i,a] = 4
        
            elif data.iloc[i,a]>=0.004:
                data.iloc[i,a] = 5
                
    # for 2-class classification
    if class_num==2:
        for i in range(data.shape[0]):
            if data.iloc[i,a]==0.0001:
                data.iloc[i,a] = 0
    
            elif data.iloc[i,a]>0.0001 and data.iloc[i,a]<0.0013:
                data.iloc[i,a] = 1
        
            elif data.iloc[i,a]>=0.0013:
                data.iloc[i,a] = 2
    
    return data