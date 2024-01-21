import numpy as np
from optparse import OptionParser
import pickle
import re
import random
from scipy.signal import hilbert
from scipy.signal import resample
from scipy.signal import argrelextrema
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
import fluence_calc
import nf_fit
import pulse_fit
import transformations as trans
#import atmos_original as atmos
import atm_parameters as atmos
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd
import scipy.optimize as opt
import glob
from scipy.optimize import minimize
import sys
import signal_interpolation_Fourier as sigF
import interpolation_Fourier as interp


############################### here we have a function to get random core values ################################################
def get_core(R):
    centerX=0
    centerY=0
    r = R * np.sqrt(random.randint(0,1000)/1000.)
    theta = random.randint(0,1000)/1000.*2*np.pi
    #core_x = 3
    #core_y = 3
    core_x = centerX + r * np.cos(theta)
    core_y = centerY + r * np.sin(theta)
    return (core_x,core_y)
    
############################## we will add some major event details  ####################################   
events = ['000014','000015','000047','000059','000066','000101','000100','000102']
#events = ['000100']
zens=[]
azs=[]
xmax_info = []
for event in events:
	path0 = glob.glob("/vol/astro7/lofar/krishna/ret/simulate/events/{0}/proton/*.long".format(event))
	print("running event {0}".format(event))
	xmaxs=[]
	showerno=[]
	phi_0=[]
	interpolated_rbf_m = []
	interpolated_rbf_fluence = []
	signal_interpolater = []
	for i in range(len(path0)):
	    f=path0[i]
	    hillas = np.genfromtxt(re.findall("PARAMETERS.*",open(f,'r').read()))[2:]
	    xmaxs.append(hillas[2])
	    showerno.append(os.path.splitext(path0[i])[0][-6:])
	iter =0
	Binc=1.4158
	h = 3200.0
	print(showerno)
	for fileno in showerno:
		print('fileno',fileno)
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
		az_rot=3*np.pi/2+azimuth
		zen_rot = zenith
	zens.append(zen_rot)
	azs.append(az_rot)
	df = pd.read_pickle('/vol/astro7/lofar/krishna/ret/simulate/event_library/xmax_info_{0}.pkl'.format(event))
	xmax_info.append(df)
	
################### all the functions to minimize ##############################################

def radio_fitfunc_f(f,dpos,n,xoff,yoff,az,zen,event_i):
    pos_ant_UVW = trans.GetUVW(dpos, xoff, yoff, zen, az,Binc,h)
    interp_power = xmax_info[event_i]['rbf_fluence'][n](pos_ant_UVW[:,0],pos_ant_UVW[:,1],0)
    return f*interp_power
    
def radio_fitfunc_m(dpos,n,xoff,yoff,az,zen,event_i):
    pos_ant_UVW = trans.GetUVW(dpos, xoff, yoff, zen, az,Binc,h)
    interp_power = xmax_info[event_i]['rbf_m'][n](pos_ant_UVW[:,0],pos_ant_UVW[:,1],0)
    return interp_power

radio_errfunc_m = lambda p,dtotm,dsigmam,weights,loc_stations,cx, cy, az, zen, n,event_i: np.mean(np.square(radio_fitfunc_m(loc_station,n,cx+p[0],cy+p[1],az,zen,event_i)/dtotm - 1))

radio_errfunc = lambda p,fluence_pow,fluence_err,weights,loc_stations,cx, cy, az, zen, n,event_i: np.mean(np.square((radio_fitfunc_f(p[0],loc_station,n,cx+p[1],cy+p[2],az,zen,event_i) - fluence_pow)/fluence_err))

def FitRoutine_radio_m(fit_args,nant,iterations=3):
    #paramset=np.ndarray([iterations,2]) # 3 free parameters in fit
    itchi2=np.zeros(iterations)
    result = []
    for j in np.arange(iterations):
        initguess=[200*np.random.rand()-100, 200*np.random.rand()-100] # Radio power scale factor, core X, core Y
        result.append(minimize(radio_errfunc_m, initguess, args=fit_args))
        itchi2[j] = radio_errfunc_m(result[j].x,*fit_args)

    bestiter=np.argmin(itchi2)
    fitparam=result[bestiter].x
    chi2_rad=radio_errfunc_m(fitparam, *fit_args)
    
    return chi2_rad,fitparam[0], fitparam[1]

def FitRoutine_radio_min(fit_args,initguess,nant,iterations=3):
    #paramset=np.ndarray([iterations,2]) # 3 free parameters in fit
    itchi2=np.zeros(iterations)
    result = []
    print(initguess)
    for j in np.arange(iterations):
        initguess=initguess # Radio power scale factor, core X, core Y
        result.append(minimize(radio_errfunc, initguess, args=fit_args))
        itchi2[j] = radio_errfunc(result[j].x,*fit_args)

    bestiter=np.argmin(itchi2)
    fitparam=result[bestiter].x
    chi2_rad=radio_errfunc(fitparam, *fit_args)
    
    return chi2_rad,fitparam[0],fitparam[1], fitparam[2]
    
############################### Minimization ##################################################################### 

x_true = []
y_true = []
xcorem_seq = []
ycorem_seq = []
fr_seq = []
xdiff=[]
ydiff = []
loc_station = np.genfromtxt('ret_I.txt')
print('hoi',loc_station)
weights = np.ones(len(loc_station))
print(range(len(event)))
for event_i in range(len(events)):
    nsim = len(xmax_info[event_i])
    print(nsim)
    print('running event',events[event_i])
    for sim_event in range(nsim):
        print(sim_event)
        for n_cores in range(100):
            core_x,core_y = get_core(100)
            core_ix,core_iy = core_x,core_y
            print(n_cores,core_ix,core_iy)
            nant = len(loc_station)
            radiochi2=np.zeros(nsim)
            p_ratio=np.zeros(nsim)
            xoffset=np.zeros(nsim)
            yoffset=np.zeros(nsim)
            pos_ant_UVW = trans.GetUVW(loc_station, core_x,core_y, zens[event_i], azs[event_i],Binc,h)
            dtotm = xmax_info[event_i]['rbf_m'][sim_event](pos_ant_UVW[:,0],pos_ant_UVW[:,1],0)
            dsigmam = dtotm*0.08
            dtotpower = xmax_info[event_i]['rbf_fluence'][sim_event](pos_ant_UVW[:,0],pos_ant_UVW[:,1],0)
            dsigmatot=dtotpower*0.15
            
            for i in np.arange(nant):
                    dtotpower[i]=dtotpower[i]+random.gauss(0,dsigmatot[i])
                    dtotm[i]=dtotm[i]+random.gauss(0,dsigmam[i])
            check = dtotpower>5.0
            print(sum(check))
            if sum(check)>=3:
                Desired_FitRoutine = FitRoutine_radio_m
                niterations=3
                for i in np.arange(nsim): # Loop over nsim, i.e. number of simulated showers. Do fit with core (x,y) as free parameters, to interpolated LDF footprint of this shower. Get chi^2, repeat for all showers.
                    #print('Simulated shower %d out of %d' % (i, nsim))
                    fit_args=(dtotm,dsigmam,weights,loc_station,core_x,core_y,azs[event_i],zens[event_i],i,event_i)
                    radiochi2[i], xoffset[i], yoffset[i] = Desired_FitRoutine(fit_args,nant,niterations)
                    #print(i,radiochi2[i],core_x+xoffset[i],core_y+yoffset[i])

                Desired_Chi2 = radiochi2
                radiochi2[sim_event]=1000
                bestsim=np.argmin(radiochi2)
                xoffm,yoffm = xoffset[bestsim],yoffset[bestsim]
                cx,cy = core_x+xoffset[bestsim],core_y+yoffset[bestsim]
                
                print('m-method-over')
                print(xoffset[bestsim],yoffset[bestsim])

                ##start minimizing with fluence again.................
                
                radiochi2=np.zeros(nsim)
                p_ratio=np.zeros(nsim)
                xoffset=np.zeros(nsim)
                yoffset=np.zeros(nsim)
                Desired_FitRoutine = FitRoutine_radio_min
                pos_ant_UVW = trans.GetUVW(loc_station, cx,cy, zens[event_i], azs[event_i],Binc,h)
                dtotm = xmax_info[event_i]['rbf_m'][sim_event](pos_ant_UVW[:,0],pos_ant_UVW[:,1],0)
                dsigmam = dtotm*0.08
                dtotpower = xmax_info[event_i]['rbf_fluence'][sim_event](pos_ant_UVW[:,0],pos_ant_UVW[:,1],0)
                dsigmatot=dtotpower*0.15
            
                for i in np.arange(nant):
                    dtotpower[i]=dtotpower[i]+random.gauss(0,dsigmatot[i])
                    dtotm[i]=dtotm[i]+random.gauss(0,dsigmam[i])
                check1 = dtotpower>5.0
                if sum(check1)>=3:
                    for i in np.arange(nsim): # Loop over nsim, i.e. number of simulated showers. Do fit with core (x,y) as free parameters, to interpolated LDF footprint of this shower. Get chi^2, repeat for all showers.
                        initguess = [1.0,xoffm,yoffm]
                        fit_args=(dtotpower,dsigmatot,weights,loc_station,cx,cy,azs[event_i],zens[event_i],i,event_i)
                        radiochi2[i],p_ratio[i],xoffset[i], yoffset[i] = Desired_FitRoutine(fit_args,initguess,nant,niterations)
                        #print(i,radiochi2[i],xoffset[i],yoffset[i])

                    Desired_Chi2 = radiochi2
                    radiochi2[sim_event]=1000
                    bestsim=np.argmin(radiochi2)
                    xcore = xoffset[bestsim]+core_ix
                    ycore = yoffset[bestsim]+core_iy
                    f_r = p_ratio[bestsim]
                    x_true.append(core_ix)
                    y_true.append(core_iy)
                    xcorem_seq.append(xcore)
                    ycorem_seq.append(ycore)
                    xdiff.append(xoffset[bestsim])
                    ydiff.append(yoffset[bestsim])
                    print('final',xcore,ycore)
                    fr_seq.append(p_ratio[bestsim])
                    print(x_true,y_true,xcorem_seq, ycorem_seq,xdiff,ydiff)
combined = np.column_stack((x_true,y_true,xcorem_seq, ycorem_seq,xdiff,ydiff,fr_seq))
np.savetxt('sequential.txt', combined)
	    

