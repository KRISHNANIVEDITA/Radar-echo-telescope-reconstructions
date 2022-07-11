import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import math

from decimal import *
from matplotlib.ticker import StrMethodFormatter
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('/vol/astro7/lofar/krishna/ret/results/flu_cut/event000027_flucut.pdf')
for flu_cut in range(1,10):
	id_var = []
	xcore_nosig = []
	ycore_nosig = []
	xcore_fit = []
	ycore_fit = []
	xcore_nofit = []
	ycore_nofit = []
	xcore = []
	ycore = []
	fluence = []
	filesUB = glob.glob("/vol/astro7/lofar/krishna/ret/data/27/*.txt")
	fluencefile = glob.glob("/vol/astro7/lofar/krishna/ret/fluence/27/*.txt")
	
	array_file='array_5a_1.txt'
	positions=np.genfromtxt(array_file,skip_header=1)
	count = 0
	count1 = 0
	for file in fluencefile:
		no = sum(1 for line in open(file))
		with open(file, 'rt') as f:
			lines = f.readlines()
			for x in lines:
				fluence.append(x.split(' '))
	ant_cut = np.ones(no)
	
	for i in range(0,no):
		z=0
		for j in range(0,6):
			if float(fluence[i][j])<flu_cut:
				z=z+1	
		if z > 3.0:
			ant_cut[i]=0
			 
	for file in filesUB:
		print(file)
		with open(file, 'rt') as f:
			lines = f.readlines()
			for x in lines:
				energy = x.split()[1]
				zenith = x.split()[2]
				id_var = np.append(id_var,x.split()[13])
				xcore = np.append(xcore,x.split()[9])
				ycore = np.append(ycore,x.split()[10])


	for i in range(0,no):
		if id_var[i] == '-1' or ant_cut[i] == 0.0:
			xcore_nosig = np.append(xcore_nosig,xcore[i])
			ycore_nosig = np.append(ycore_nosig,ycore[i])
		if id_var[i] == '1' and ant_cut[i] == 1.0 :
			xcore_fit = np.append(xcore_fit,xcore[i])
			ycore_fit = np.append(ycore_fit,ycore[i])
		if id_var[i] == '2':
			xcore_nofit = np.append(xcore_nofit,xcore[i])
			ycore_nofit = np.append(ycore_nofit,ycore[i])
	'''
	for i in range(0,6000):
		if id_var[i] == '1' or id_var[i] == '2' or id_var[i] == '0' :
			count = count+1
		if(ant_cut[i] == 1.0):
			count1 = count1 + 1
		
	print(count,"\t",count1)	
	'''
	print(len(xcore_fit))	
	xcore_nosig = [float(x) for x in xcore_nosig]
	ycore_nosig = [float(x) for x in ycore_nosig]
	xcore_fit = [float(l) for l in xcore_fit]
	ycore_fit = [float(l) for l in ycore_fit]
	xcore_nofit = [float(l) for l in xcore_nofit]
	ycore_nofit = [float(l) for l in ycore_nofit]

	fig = plt.figure(figsize=(7,7))
	ax1 = fig.add_subplot(111)
	#ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
	ax1.scatter(xcore_nosig,ycore_nosig, s = 10, c = 'b',alpha = 0.3, label = 'no signal')
	ax1.scatter(xcore_fit,ycore_fit, s = 10, c = 'yellow', alpha = 0.4,label = 'fit converged')
	ax1.scatter(xcore_nofit,ycore_nofit, s = 10, c = 'r', alpha = 0.5,label = 'fit didnt converge')
	ax1.scatter(positions[:,0],positions[:,1], s = 20, c = 'black',label = 'antenna stations',marker = 'x')
	ax1.set_title('event = 000027, Energy = {0}, Zenith = {1:.2f}, flu_cut = {2}'.format(energy,float(zenith)*180/math.pi,flu_cut))
	plt.legend()
	pp.savefig()
pp.close()			

