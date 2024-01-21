import numpy as np
from optparse import OptionParser
import pickle
import re
import random
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
import import_ipynb
import fluence_calc
import nf_fit
from scipy.stats import norm
import matplotlib.mlab as mlab
import pulse_fit
import transformations as trans
#import atmos_original as atmos
import atm_parameters as atmos
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd
import scipy.optimize as opt
import random_antenna_fit as random_fit
import signal_interpolation_Fourier as sigF
import interpolation_Fourier as interp
import glob
import scipy.stats
import numpy as np
import pandas as pd
import seaborn as sns

from fitter import Fitter, get_common_distributions, get_distributions


def fit_function(x, mu, sigma):
    return np.exp(-0.5*((x-mu)/sigma)**2)/np.sqrt(2*np.pi)*sigma
def Gauss(x, A, B,c): 
    y = A*np.exp(-1*B*(x-c)**2) 
    return y 


###################### to plot with fitting in the histogram ###############################################

file_path1 = '/vol/astro7/lofar/krishna/edit_final/m_only.txt'
file_path2 = '/vol/astro7/lofar/krishna/edit_final/fluence_interp.txt'
from matplotlib.patches import Rectangle
#file_path3 = '/vol/astro7/lofar/krishna/edit_final/keepsafe/ret_seqI.csv'
# Initialize lists to store the data for each column
column_arrays1 = []
column_arrays2 = []
column_arrays3 = []

# Read the file and process each line
with open(file_path1, 'r') as file:
    lines = file.readlines()
    # Remove the first line (header)
    lines = lines[1:]    
    for line in lines:
        # Split the line by the delimiter (comma in this case)
        columns = line.strip().split(' ' )        
        # Convert each column to a float array
        column_array = np.array([float(col) for col in columns])        
        # Append the array to the list
        column_arrays1.append(column_array)

# Read the file and process each line
with open(file_path2, 'r') as file:
    lines = file.readlines()
    # Remove the first line (header)
    lines = lines[1:]    
    for line in lines:
        # Split the line by the delimiter (comma in this case)
        columns = line.strip().split(' ')        
        # Convert each column to a float array
        column_array = np.array([float(col) for col in columns])        
        # Append the array to the list
        column_arrays2.append(column_array)
        

'''
# Read the file and process each line
with open(file_path3, 'r') as file:
    lines = file.readlines()
    # Remove the first line (header)
    lines = lines[1:]    
    for line in lines:
        # Split the line by the delimiter (comma in this case)
        columns = line.strip().split(',')        
        # Convert each column to a float array
        column_array = np.array([float(col) for col in columns])        
        # Append the array to the list
        column_arrays3.append(column_array)
'''
xcore1 = []
ycore1 = []
#xmaxdiff1 = []
xcore2 = []
ycore2 = []
#xmaxdiff2 = []
xcore3 = []
ycore3 = []
#xmaxdiff3 = []
f_r1 = []
f_r2=[]
for i in range(len(column_arrays2)):
    xcore1.append(column_arrays1[i][4])
    ycore1.append(column_arrays1[i][5])
    #f_r1.append(column_arrays1[i][4])
    #xmaxdiff1.append(column_arrays1[i][3])
for i in range(len(column_arrays2)):
    xcore2.append(column_arrays2[i][4])
    ycore2.append(column_arrays2[i][5])
    f_r2.append(column_arrays2[i][6])
    #xmaxdiff2.append(column_arrays2[i][3])
print(len(xcore1),len(xcore2))    
'''
for i in range(len(column_arrays2)):
    xcore3.append(column_arrays3[i][1])
    ycore3.append(column_arrays3[i][2])
    #xmaxdiff3.append(column_arrays3[i][3])
'''
bins_x =5000
bins_y = 500
bins_x = np.histogram(np.hstack((xcore1, xcore2)), bins=bins_x)[1]
bins_y = np.histogram(np.hstack((ycore1, ycore2)), bins=bins_y)[1]
#bins_fr = np.histogram(np.hstack((f_r1, f_r2)), bins=5000)[1]

#factor = np.sum(hist)
xhist1, binedgesx1 = np.histogram(xcore1,bins=2000)
xhist2, binedgesx2 = np.histogram(xcore2,bins=2000)
binwx1 = binedgesx1[1]-binedgesx1[0]
binwx2 = binedgesx2[1]-binedgesx2[0]
bincentersx1 = np.mean(np.vstack([binedgesx1[0:-1],binedgesx1[1:]]), axis=0)
bincentersx2 = np.mean(np.vstack([binedgesx2[0:-1],binedgesx2[1:]]), axis=0)
totalx1 = np.sum(xhist1)
totalx2 = np.sum(xhist2)
paramsx1, covariance = curve_fit(Gauss, bincentersx1, xhist1/(totalx1*binwx1), p0=[0, 1,0])
paramsx2, covariance = curve_fit(Gauss, bincentersx2, xhist2/(totalx2*binwx2), p0=[0, 1,0])
print('mean=',paramsx1[2],'sigma=',np.sqrt(1/(2*paramsx1[1])))
print('mean=',paramsx2[2],'sigma=',np.sqrt(1/(2*paramsx2[1])))
#fig = plt.figure(1,figsize=(20,20))
#ax1 = fig.add_subplot(3,1,1)
#ax2 = fig.add_subplot(3,1,2)
#ax3 = fig.add_subplot(3,1,3)
#plt.hist(xcore2,color='gold',alpha=0.5,bins =2000,label='6station',density=True)
plt.hist(xcore2,color='red',alpha=0.5,bins =2000,density=True)
plt.xlim(-60,60)
plt.xlabel('true_core_x-reco_core_x (m)')
plt.ylabel('frequency')
#factor = np.sum(counts)
#ax1.hist(bin_edges[:-1], bins, weights=hist/factor)
#ax2.hist(ycore1,color='green',alpha=0.5,bins = bins_y,label='6station')
plt.plot(bincentersx2, Gauss(bincentersx2, *paramsx2), 'k--', label='Fit: mu=%0.2f, sigma=%0.2f' % (paramsx1[2], np.sqrt(1/(2*paramsx1[1]))))
plt.legend()
plt.savefig('corex.png')
#plt.plot(bincentersx2, Gauss(bincentersx2, *paramsx2), 'r--', label='Fit: mu=%0.2f, sigma=%0.2f' % (params[0], params[1]))
#ax2.plot(bins_y, fit_y1, 'r--', linewidth=2)
#ax3.hist(f_r1,color='gold',alpha=0.5,bins = 1000)
#ax1.hist(xcore2,color='gold',bins = bins_x,alpha=0.5)
#ax2.hist(ycore2,color='gold',bins = bins_y,alpha=0.5)
#ax3.hist(f_r2,color='gold',bins = 1000)
#ax1.hist(xcore3,color='red',alpha=0.4,bins = 100)
#ax2.hist(ycore2,color='black',alpha=0.4,bins = 100,label='only fluence')
#ax1.set_facecolor('black')
#ax2.set_facecolor('black')
#ax3.set_facecolor('black')
#ax1.set_xlabel('x_reco - x_true (m)')
#ax2.set_xlabel('y_reco - y_true (m)')
#ax1.set_ylabel('frequency')
#ax2.set_ylabel('frequency')
#ax3.set_xlabel('f_r ratio')
#ax3.set_ylabel('frequency')

########### simply plot and compare 2 histograms ###################################

file_path1 = '/vol/astro7/lofar/krishna/edit_final/m_only.txt'
file_path2 = '/vol/astro7/lofar/krishna/edit_final/fluence_interp.txt'
from matplotlib.patches import Rectangle
#file_path3 = '/vol/astro7/lofar/krishna/edit_final/keepsafe/ret_seqI.csv'
# Initialize lists to store the data for each column
column_arrays1 = []
column_arrays2 = []
column_arrays3 = []

# Read the file and process each line
with open(file_path1, 'r') as file:
    lines = file.readlines()
    # Remove the first line (header)
    lines = lines[1:]    
    for line in lines:
        # Split the line by the delimiter (comma in this case)
        columns = line.strip().split(' ' )        
        # Convert each column to a float array
        column_array = np.array([float(col) for col in columns])        
        # Append the array to the list
        column_arrays1.append(column_array)

# Read the file and process each line
with open(file_path2, 'r') as file:
    lines = file.readlines()
    # Remove the first line (header)
    lines = lines[1:]    
    for line in lines:
        # Split the line by the delimiter (comma in this case)
        columns = line.strip().split(' ')        
        # Convert each column to a float array
        column_array = np.array([float(col) for col in columns])        
        # Append the array to the list
        column_arrays2.append(column_array)
xcore1 = []
ycore1 = []
#xmaxdiff1 = []
xcore2 = []
ycore2 = []
#xmaxdiff2 = []
xcore3 = []
ycore3 = []
#xmaxdiff3 = []
f_r1 = []
f_r2=[]
for i in range(len(column_arrays2)):
    xcore1.append(column_arrays1[i][4])
    ycore1.append(column_arrays1[i][5])
    #f_r1.append(column_arrays1[i][4])
    #xmaxdiff1.append(column_arrays1[i][3])
for i in range(len(column_arrays2)):
    xcore2.append(column_arrays2[i][4])
    ycore2.append(column_arrays2[i][5])
    #f_r2.append(column_arrays2[i][4])
    
#count, binedges = np.histogram(xcore1,bins=1000)
#bincenters = np.mean(np.vstack([binedges[0:-1],binedges[1:]]), axis=0)
#params, covariance = curve_fit(fit_function, bincenters, yhist, p0=[0, 1])
plt.figure()
plt.hist(xcore1,color='green',alpha=0.5,bins = bins_x,label='6station')
plt.hist(xcore2,color='gold',alpha=0.5,bins = bins_x,label='6station')
#plt.plot(bincenters, fit_function(bincenters, *params), 'r--', label='Fit: mu=%0.2f, sigma=%0.2f' % (params[0], params[1]))

###### To plot a scatter plot ###########################################


file_path1 = '/vol/astro7/lofar/krishna/edit_final/fluence_interp.txt'
column_arrays1 = []
with open(file_path1, 'r') as file:
    lines = file.readlines()
    # Remove the first line (header)
    #lines = lines[1:]    
    for line in lines:
        # Split the line by the delimiter (comma in this case)
        columns = line.strip().split(' ')
        # Convert each column to a float array
        #if columns[1]=='2':
        column_array = np.array([float(col) for col in columns])        
        # Append the array to the list
        column_arrays1.append(column_array)
print(len(column_arrays1))        
X=[]
Y=[]
C=[]
xreco=[]
yreco=[]
for i in range(len(column_arrays1)):
    X.append(column_arrays1[i][0])
    Y.append(column_arrays1[i][1])
    xreco.append(column_arrays1[i][4])
    yreco.append(column_arrays1[i][5])
    C.append(np.sqrt(column_arrays1[i][4]**2+column_arrays1[i][5]**2))
print(len(X))
fig_2 = plt.figure()
ax2 = fig_2.add_subplot(1,1,1)
im=ax2.scatter(X,Y,s=2,c=C,cmap='Reds',vmin=0,vmax=10)
ax2.set_title('fluence minimization')
#ax2.scatter(X1,Y1,s=2,c='magenta',alpha=0.3,label='not triggered')
ax2.set_xlabel('X(m)')
ax2.set_ylabel('Y(m)')
ax2.legend()
cb=fig_2.colorbar(im)
cb.set_label('radius difference core')
plt.savefig('fluence.png')

