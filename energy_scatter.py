import numpy as np
from optparse import OptionParser
import pickle
import re
import glob
import random
from scipy.signal import argrelextrema
from scipy.signal import hilbert
from scipy.signal import resample
import scipy.fftpack as fftp
import os
from NuRadioReco.utilities import units
from itertools import islice
import math
import process_func as prf
import importlib as imp
import matplotlib.pyplot as plt
import fluence as flu
import scipy.interpolate as intp
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import fluence_calc
import nf_fit
import pulse_fit
import pandas as pd
import transformations as trans
from matplotlib.pyplot import cm
import sys
sys.path.insert(1,'/home/knivedita/scratch/ret_ffm/')
import signal_interpolation_fourier as sigF




def fluence_fit(mf,A,s):
    return A*np.exp(-s*np.abs(mf)**0.9) 
    
    
def findmiddle(xmaxs1):
    xmaxs1.sort()
    if len(xmaxs1)%2==0:
        return xmaxs1[int(len(xmaxs1)/2-2)], xmaxs1[int(len(xmaxs1)/2-1)],xmaxs1[int(len(xmaxs1)/2+1)],xmaxs1[int(len(xmaxs1)/2)+2]
    if len(xmaxs1)%2==1:
        return xmaxs1[int((len(xmaxs1)-1)/2 -1)],xmaxs1[int((len(xmaxs1)-1)/2)],xmaxs1[int((len(xmaxs1)+1)/2)],xmaxs1[int((len(xmaxs1)+1)/2 + 1)]
        
        
events_25 = ['000036','000037','000038','000039']
events_30 = ['000011','000012','000013','000014','000015','000016']
events_33 = ['000064','000065','000066','000067']
events_35 = ['000040','000041','000042','000043']

events = events_25+events_30+events_33+events_35
nf = []
m = []
m_in = []
m_out=[]
phi_in=[]
phi_out=[]
fluence = []
phi=[]
m_cut=[]
nf_cut=[]
fluence_cut=[]
phi_cut=[]

for event in events:
    print(event)
    path0 = glob.glob("/vol/astro7/lofar/krishna/ret/simulate/events/{0}/proton/*.long".format(event))
    print("running event {0}".format(event))
    xmaxs=[]
    showerno=[]
    for i in range(len(path0)):
        f=path0[i]
        hillas = np.genfromtxt(re.findall("PARAMETERS.*",open(f,'r').read()))[2:]
        xmaxs.append(hillas[2])
        showerno.append(os.path.splitext(path0[i])[0][-6:])
    for fileno in showerno:
        print("running shower",fileno)
        #define basic directories and file routes
        datadir='/vol/astro7/lofar/krishna/ret/simulate/events/{0}/proton/'.format(str(event))
        steerfile = '{0}/steering/RUN{1}.inp'.format(datadir,str(fileno).zfill(6))
        listfile = open('{0}/steering/SIM{1}.list'.format(datadir,str(fileno).zfill(6)))
        lorafile = '{0}/DAT{1}.lora'.format(datadir,str(fileno).zfill(6))
        longfile = '{0}/DAT{1}.long'.format(datadir,str(fileno).zfill(6))
        longdata=np.genfromtxt(longfile, skip_header=2, skip_footer=5, usecols=(0,2,3))

        #extract xmax,zenith,energies from these files
        xlength=np.argmax(np.isnan(longdata[:,0]))
        Xground=xlength*10.0
        profile = longdata[0:xlength,:]
        hillas = np.genfromtxt(re.findall("PARAMETERS.*",open(longfile,'r').read()))[2:]
        zenith=(np.genfromtxt(re.findall("THETAP.*",open(steerfile,'r').read()))[1])*np.pi/180. #rad; CORSIKA coordinates
        azimuth=np.mod(np.genfromtxt(re.findall("PHIP.*",open(steerfile,'r').read()))[1],360)*np.pi/180.  #rad; CORSIKA coordinates
        energy=np.genfromtxt(re.findall("ERANGE.*",open(steerfile,'r').read()))[1] #GeV
        lines =  listfile.readlines()
        xmax=hillas[2]
        az_rot=3*np.pi/2+azimuth
        zen_rot = zenith
        print(zenith*180/np.pi,xmax,energy)
        #defining all necessary variable
        nantennas=160 
        antenna_position=np.zeros([nantennas,3])
        f = []
        E = []
        fit = []
        Binc=1.4158
        h = 3200.0
        perp_ant = np.zeros([40,3])
        perp_file = []
        antenna_file = []
        r2 = np.zeros(160)

        #get_antenna_positions
        index=0
        for j in np.arange(nantennas):
                antenna_file.append(lines[j].split(" ")[5])
                antenna_position[j] = (lines[j].split(" ")[2:5])
        antenna_position[:,0] = [x/100.0 for x in antenna_position[:,0]]
        antenna_position[:,1] = [x/100.0 for x in antenna_position[:,1]]
        antenna_position[:,2] = [x/100.0  for x in antenna_position[:,2]]        
        antenna_file =  [s.strip() for s in antenna_file]
        antenna_right=np.zeros([nantennas,3])
        antenna_right[:,0], antenna_right[:,1], antenna_right[:,2] = -antenna_position[:,1], antenna_position[:,0], antenna_position[:,2]
        antUVW=trans.GetUVW(antenna_right, 0, 0, zen_rot, az_rot,Binc,h)

        for k in np.arange(nantennas):
                if k%2 == 0 and k%4 != 0:
                    perp_ant[index] = antenna_right[k]
                    perp_file.append(lines[k].split(" ")[5])
                    index = index + 1
        perp_file =  [s.strip() for s in perp_file]
        antUVW_perp = trans.GetUVW(perp_ant,0,0,zen_rot,az_rot,Binc,h)


        #get_electric_fields_fit_mf_nf_returned

        for j in np.arange(0,160):
            coreasfile = '{0}/SIM{1}_coreas/raw_{2}.dat'.format(datadir,str(fileno).zfill(6),antenna_file[j])
            f1,E1 = pulse_fit.pulse_cut(coreasfile,azimuth,zenith,Binc,h)
            m_shower,nf_shower,fit1,pcov = pulse_fit.nf_fit2(f1,E1,azimuth,zenith,Binc,h)
            #print("Cut the pulse and fitting-antenna{0}".format(antenna_file[j]))
            fluence_shower_antenna = fluence_calc.fluence_calc(coreasfile,azimuth,zenith)
            if fluence_shower_antenna > 0.0 and m_shower<0 :
                phi_shower = math.sqrt(fluence_shower_antenna)/(10**(round(math.log10(energy),2)+9))
                m.append(m_shower)
                nf.append(nf_shower)
                fluence.append(fluence_shower_antenna)
                phi.append(phi_shower)
m=[-x*10**3 for x in m]

df1 = pd.DataFrame({"nf": nf,"mf": m, "phi":phi,'fluence':fluence}) 
#df1.to_csv("m_phi.csv", index=False)

m_in = df1.loc[df1["nf"]<0,"mf"].tolist()
phi_in = df1.loc[df1["nf"]<0,"phi"].tolist()
m_out = df1.loc[df1["nf"]>0,"mf"].tolist()
phi_out = df1.loc[df1["nf"]>0,"phi"].tolist()
w1 = np.ones_like(phi_in)
w2 = np.ones_like(phi_out)

df_in = pd.DataFrame({"m_in": m_in,"phi_in":phi_in}) 
df_out = pd.DataFrame({"m_out": m_out,"phi_out":phi_out})

df_in.to_pickle('df_in.pkl')
df_out.to_pickle('df_out.pkl')

