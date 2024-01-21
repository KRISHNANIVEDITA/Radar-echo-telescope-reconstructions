#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from optparse import OptionParser
import pickle
import re
import glob
from scipy.signal import hilbert
from scipy.signal import resample
import scipy.fftpack as fftp
import os
from itertools import islice
import math
import process_func as prf
import importlib as imp
import matplotlib.pyplot as plt
import fluence as flu
import scipy.interpolate as intp
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import fluence as flu
import nf_fit
import pulse_fit
import pandas as pd
import transformations as trans
import atm_parameters as atmos
from matplotlib.pyplot import cm
import signal_interpolation_Fourier as sigF
from scipy.signal import argrelextrema
import random


# In[2]:


def fluence_calc_here(time,interpolated_pulse):
    hold=flu.calculate_energy_fluence_vector(interpolated_pulse,time, signal_window=100., remove_noise=True)
    fluence_data=hold[0]+hold[1]
    return fluence_data


# In[3]:


def reco_energy(A,s,mf,flu):
    return flu/(A*np.exp(-s*np.abs(mf)**0.8))


# In[13]:


def energy_reconstruction(m1,nf,A1,A2,s1,s2,fluence_data):
    #A1,A2,s1,s2=1.9687558344864574e-17,3.7274977210235024e-17,-0.02170087127996101,.0000000026919538
    #m1,nf=-0.002565016642487396,-1.1183155482864325e-05
    #fluence_data=119
    if nf<=0:
        #print("nf<0")
        #print("hello")
        flu=np.sqrt(fluence_data)
        #print(flu)
        mf = m1*10**3
        Energy = reco_energy(A1,s1,mf,flu)
    if nf>0:
        #print("nf>0")
        flu=np.sqrt(fluence_data)
        Energy = reco_energy(A2,s2,m1*10**3,flu)
    return Energy


# In[5]:


def random_xy(radius):
        r = radius * math.sqrt(random.uniform(0, 1))
        theta = random.uniform(0,1) * 2 * np.pi
        return r * math.cos(theta),r * math.sin(theta)


# In[6]:


def get_antenna_SP(listfile,nantennas,zen_rot,az_rot,Binc,h):
    lines = listfile.readlines()
    antenna_position=np.zeros([nantennas,3])
    antenna_file = []
    for j in np.arange(nantennas):
                    antenna_position[j] = (lines[j].split(" ")[2:5])
                    antenna_file.append(lines[j].split(" ")[5])
    antenna_file =  [s.strip() for s in antenna_file]
    antenna_position[:,0] = [x/100.0 for x in antenna_position[:,0]]
    antenna_position[:,1] = [x/100.0 for x in antenna_position[:,1]]
    antenna_position[:,2] = [x/100.0  for x in antenna_position[:,2]]        
    antenna_right=np.zeros([nantennas,3])
    antenna_right[:,0], antenna_right[:,1], antenna_right[:,2] = -antenna_position[:,1], antenna_position[:,0], antenna_position[:,2]
    antUVW=trans.GetUVW(antenna_right, 0, 0, zen_rot, az_rot,Binc,h)
    return antUVW,antenna_file


# In[7]:


def Exy(event_data,shower_data):
    #make into one matrix and interpolate
    Binc=1.4158
    h = 3200.0
    datadir='/vol/astro7/lofar/krishna/ret/simulate/events/{0}/proton/'.format(str(event_data))
    
    #datadir='/vol/astro7/lofar/sim/pipeline/events/{0}/0/coreas/proton/'.format(str(event_data))
    #Binc=1.1837
    #h = 7.6
    nantennas=160
    #get_zenith and azimuth
    steerfile = '{0}/steering/RUN{1}.inp'.format(datadir,str(shower_data).zfill(6))
    zenith=(np.genfromtxt(re.findall("THETAP.*",open(steerfile,'r').read()))[1])*np.pi/180. #rad; CORSIKA coordinates
    azimuth=np.mod(np.genfromtxt(re.findall("PHIP.*",open(steerfile,'r').read()))[1],360)*np.pi/180.  #rad; CORSIKA coordinates
    energy=np.genfromtxt(re.findall("ERANGE.*",open(steerfile,'r').read()))[1] #GeV
    az_rot=3*np.pi/2+azimuth
    zen_rot = zenith
    ##define vector positions antenna woth Ex,y,z
    listfile = open('{0}/steering/SIM{1}.list'.format(datadir,str(shower_data).zfill(6)))
    antUVW,antenna_file = get_antenna_SP(listfile,nantennas,zen_rot,az_rot,Binc,h)  
    coreasfile = '{0}/SIM{1}_coreas/raw_{2}.dat'.format(datadir,str(shower_data).zfill(6),antenna_file[0])
    data=np.genfromtxt(coreasfile)
    dlength=int(data.shape[0])
    poldata=np.ndarray([160,dlength,2])
    poldata_filt=np.ndarray([160,dlength,2])
    
    for j in np.arange(0,160):
        coreasfile = '{0}/SIM{1}_coreas/raw_{2}.dat'.format(datadir,str(shower_data).zfill(6),antenna_file[j])
        data=np.genfromtxt(coreasfile)
        data[:,1:]*=2.99792458e4 
        if j==0:
            time = data[:,0]
            
        az_rot=3*np.pi/2+azimuth    #conversion from CORSIKA coordinates to 0=east, pi/2=north
        zen_rot=zenith 
        poldata[j,:,0] = -1.0/np.sin(zen_rot)*data[:,3] #-1/sin(theta) *z
        poldata[j,:,1] = np.sin(az_rot)*data[:,2] + np.cos(az_rot)*data[:,1]

    #signal interpolation and plotting
    

    signal_interpolator = sigF.interp2d_signal(antUVW[:,0], antUVW[:,1], poldata,lowfreq=50.0, highfreq=350.0)
    return time,signal_interpolator
    
    


# In[ ]:





# In[8]:


def main(event_data,shower_data,time,signal_interpolator,A1,A2,s1,s2):
    Binc=1.4158
    h = 3200.0
    datadir='/vol/astro7/lofar/krishna/ret/simulate/events/{0}/proton/'.format(str(event_data))
    #datadir='/vol/astro7/lofar/sim/pipeline/events/{0}/0/coreas/proton/'.format(str(event_data))
    #Binc=1.1837
    #h = 7.6
    nantennas=160

    #get_zenith and azimuth
    steerfile = '{0}/steering/RUN{1}.inp'.format(datadir,str(shower_data).zfill(6))
    zenith=(np.genfromtxt(re.findall("THETAP.*",open(steerfile,'r').read()))[1])*np.pi/180. #rad; CORSIKA coordinates
    azimuth=np.mod(np.genfromtxt(re.findall("PHIP.*",open(steerfile,'r').read()))[1],360)*np.pi/180.  #rad; CORSIKA coordinates
    energy=np.genfromtxt(re.findall("ERANGE.*",open(steerfile,'r').read()))[1] #GeV
    az_rot=3*np.pi/2+azimuth
    zen_rot = zenith
    ##define vector positions antenna woth Ex,y,z
    listfile = open('{0}/steering/SIM{1}.list'.format(datadir,str(shower_data).zfill(6)))
    antUVW,antenna_file = get_antenna_SP(listfile,nantennas,zen_rot,az_rot,Binc,h)  
    coreasfile = '{0}/SIM{1}_coreas/raw_{2}.dat'.format(datadir,str(shower_data).zfill(6),antenna_file[0])
    data=np.genfromtxt(coreasfile)
    dlength=int(data.shape[0])


    X,Y = random_xy(70)
    print(X,Y)

    interpolated_pulse = signal_interpolator(X,Y,filter_up_to_cutoff=False)
    
    fluence_data = fluence_calc_here(time,interpolated_pulse)
    print("fluence is",fluence_data)
    interpolated_fft = np.abs(np.fft.rfft(interpolated_pulse.T[1]))
    freqs = np.fft.rfftfreq(dlength, d=1.0000e-10)
    freqs /= 1e6 # in MHz
    for i in range(len(freqs)):
        if freqs[i] <= 50.0:
            fmin = i
        if freqs[i] >= 350.0:
            fmax = i-1
            break
    f = freqs[fmin+1:fmax+1]
    interpolated_fft = interpolated_fft[fmin+1:fmax+1]
    index=len(f)
    local_mins=argrelextrema(interpolated_fft, np.less)[0]
    for k in range(0,len(local_mins)):
        if(f[local_mins[k]] > 150):
            index = local_mins[k]
            break
    f1 = f[0:index]
    E1 = interpolated_fft[0:index]
    m1,nf,fit1,pcov = pulse_fit.nf_fit2(f1,E1,azimuth,zenith,Binc,h)    

    energy_reco = energy_reconstruction(m1,nf,A1,A2,s1,s2,fluence_data)
    
    return energy_reco,fluence_data



# In[10]:
 #datadir='/vol/astro7/lofar/sim/pipeline/events/{0}/0/coreas/proton/'.format(str(event_data))
    #Binc=1.1837
    #h = 7.6
    #nantennas=160




# In[ ]:





# In[ ]:




