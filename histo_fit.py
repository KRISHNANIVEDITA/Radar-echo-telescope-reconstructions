import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
from decimal import *
from matplotlib.ticker import StrMethodFormatter
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages

# Importing Pandas as pd
import pandas as pd
def gaussian(x, a, mean, sigma):
    return a * np.exp(-((x - mean)**2 / (2 * sigma**2)))

filesUB = glob.glob("/vol/astro7/lofar/krishna/ret/data/efiles/*/*.txt")
filesUB = sorted(filesUB)
pp = PdfPages('/vol/astro7/lofar/krishna/ret/results/fit/fr.pdf')
pp1 = PdfPages('/vol/astro7/lofar/krishna/ret/results/fit/offset.pdf')



rows1 = []
rows2 = []


for file in filesUB:
	convergeUB = []
	delxmax_unfineUB = []
	fr_unfineUB = []
	delxmaxUB = [] 
	frUB = []
	frlogUB = []
	xoff_unfineUB = []
	xoffUB = []
	yoff_unfineUB = []
	yoffUB = []
	energy = []
	zenith = []
	print(file)
	with open(file, 'rt') as f:
		lines = f.readlines()[1:]
		for x in lines:
			energy = np.append(energy,x.split()[1])
			zenith = np.append(zenith,x.split()[2])
			convergeUB = np.append(convergeUB,x.split()[8])
			#delxmax_unfineUB = np.append(delxmax_unfineUB,x.split()[7])
			fr_unfineUB = np.append(fr_unfineUB,x.split()[4])
			xoff_unfineUB = np.append(xoff_unfineUB,x.split()[11])
			yoff_unfineUB = np.append(yoff_unfineUB,x.split()[12]) 
			
	for i in range(len(convergeUB)):
		if convergeUB[i] == '1':
			#delxmaxUB = np.append(delxmaxUB,delxmax_unfineUB[i])
			frUB = np.append(frUB,fr_unfineUB[i])
			xoffUB = np.append(xoffUB,xoff_unfineUB[i])
			yoffUB = np.append(yoffUB,yoff_unfineUB[i])
	#print(xoffUB[1:100])
	fr1UB = [float(k) for k in frUB]

	xoff1UB = [float(x) for x in xoffUB]

	yoff1UB = [float(y) for y in yoffUB]

	###########################fit the histo with gaussian###############
	mu, std = norm.fit(fr1UB)
	fr1UB.sort()

	x_axis=np.linspace(-1,3,300)
	bins_xmax=np.linspace(-1,3,300)

	e=float(energy[0])
	z = float(zenith[0])



	plt.figure(1)
	(n, bins, patches) = plt.hist(fr1UB, bins_xmax, histtype='step',color='red',alpha = 0.8,label='histogram_fr')
	maxcount = max(n)
	bins2=bins[:-1]+((bins[2]-bins[1])/2)
	popt, pcov = curve_fit(gaussian, bins2,n, p0 = [100, 1.0, 0.5])
	plt.plot(x_axis, gaussian(x_axis, *popt),color='orange',label='fit')
	plt.legend()
	plt.xlabel("fr")
	plt.ylabel("No.of Simulations")
	plt.title('Energy = 10^{0:.2f}, Zenith = {1:.2f}'.format(math.log10(e*10**9),z*180/math.pi))
	plt.text(-0.7, maxcount/3, 'mean = {0:.3f}\nstd = {1:.3f}'.format(popt[1],popt[2]), fontsize = 10)
	#plt.title('Energy = {0}, Zenith = {1:.2f}, flu_cut = {2}'.format(energy,float(zenith)*180/math.pi,flu_cut))
	pp.savefig()
	plt.close()
	rows1.append(['10^{0:.2f}'.format(math.log10(e*10**9)), '{0:.2f}'.format(z*180/math.pi), '{0:.3f}'.format(popt[1]), '{0:.3f}'.format(popt[2])])
	x_axis1=np.linspace(-80,80,300)
	bins_xmax1=np.linspace(-80,80,300)

	plt.figure(2)
	(n, bins1, patches) = plt.hist(xoff1UB, bins_xmax1, histtype='step',color='black',alpha=0.8,label='histogram_xoffset')
	maxcount = max(n)
	bins21=bins1[:-1]+((bins1[2]-bins1[1])/2)
	popt1, pcov1 = curve_fit(gaussian, bins21,n, p0 = [100, 1.0, 0.5])
	plt.plot(x_axis1, gaussian(x_axis1, *popt1),color = 'green',alpha=0.8,label='fit')
	plt.legend()
	plt.text(-70, maxcount/3, 'mean = {0:.3f}\nstd = {1:.3f}'.format(popt1[1],popt1[2]), fontsize = 10)
	plt.xlabel("Offset_xcore")
	plt.ylabel("No.of Simulations")
	plt.title('Energy = 10^{0:.2f}, Zenith = {1:.2f}'.format(math.log10(e*10**9),z*180/math.pi))
	rows2.append(['10^{0:.2f}'.format(math.log10(e*10**9)), '{0:.2f}'.format(z*180/math.pi), '{0:.3f}'.format(popt1[1]), '{0:.3f}'.format(popt1[2])])
	pp1.savefig()
	plt.close()
pp.close()
pp1.close()



offset = pd.DataFrame(rows2,columns =['energy', 'zenith', 'mean','deviation'])
f_r = pd.DataFrame(rows1,columns =['energy', 'zenith', 'mean','deviation'])

fig, ax = plt.subplots()

# hide axes
fig.patch.set_visible(True)
ax.axis('off')
ax.axis('tight')
ax.set_title("E_reco/E_true")
ax.table(cellText=f_r.values, colLabels=offset.columns, loc='center')

fig.tight_layout()
plt.savefig('/vol/astro7/lofar/krishna/ret/results/fit/fitf_r.png')

fig, ax = plt.subplots()

# hide axes
fig.patch.set_visible(True)
ax.axis('off')
ax.axis('tight')
ax.table(cellText=offset.values, colLabels=offset.columns, loc='center')
ax.set_title("xaxis_core position offset")
fig.tight_layout()
plt.savefig('/vol/astro7/lofar/krishna/ret/results/fit/fitoffset.png')





'''


counts, bins, bars = plt.hist(xoff1UB, bins=500, normed=True, alpha=0.6, color='blue',label = "50-350MHz")
print(counts)
# Plot the PDF.
p = norm.pdf(xoff1UB, mu, std)
print(p[100])
plt.plot(xoff1UB, p, 'k', linewidth=2)  
print("1")
plt.legend()
plt.xlabel("Offset_core")
plt.ylabel("No.of Simulations")
plt.savefig("/vol/astro7/lofar/krishna/ret/scripts/pics/fit.png")
plt.close()




























#plt.savefig("/vol/astro7/lofar/krishna/ret/scripts/pics/offset.png")


plt.plot(x, p, 'k', linewidth=2)
plt.xlim(-100.0,100.0)
plt.hist(np.asarray(xoff1UB, dtype='float'), normed = True, color = 'red', bins=500,label = "50-350MHz")
plt.plot(xoff1UB, norm.pdf(xoff1UB, mu, std),'k',linewidth=2)



plt.figure(1)
plt.xlim(-1, 150)
plt.hist(np.asarray(delxmaxUB, dtype='float'), color = "green", bins=200,alpha = 0.4,label = "50-350MHz")
print("1")
plt.legend()
plt.savefig("/vol/astro7/lofar/krishna/ret/scripts/pics/delxmax.png")
plt.close()

plt.figure(1)
plt.xlim(-1.0, 3.0)
plt.hist(np.asarray(fr1UB, dtype='float'), histtype=u'step', density=True,color = 'red', bins=500,label = "50-350MHz")
print("1")
plt.legend()
plt.xlabel("fr")
plt.ylabel("No.of Simulations")
plt.savefig("/vol/astro7/lofar/krishna/ret/scripts/pics/fr.png")
plt.close()

fig = plt.figure(4)
ax1 = fig.add_subplot(111)
ax1.scatter(xoff1UB,yoff1UB, s = 10, c = 'r', label = 'offset2d')
ax1.set_xlim(-100,100)
ax1.set_ylim(-100,100)
ax1.set_xlabel("Xoffset")
ax1.set_ylabel("Yoffset")
plt.savefig("/vol/astro7/lofar/krishna/ret/scripts/pics/offset2d.png")
plt.close()
'''
