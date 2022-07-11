import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.ticker as mtick
import re
import long_shower
import shutil
from decimal import *

from matplotlib.backends.backend_pdf import PdfPages
def findMiddle(xmaxs):
	xmaxs.sort()
	#print(xmax)
	middle = float(len(xmaxs))/2
	if middle % 2 != 0:
		return xmaxs[int(middle - .5)]
	else:
		return (xmaxs[int(middle)])
avg = []
xmaxs = []
xmax_fileorder = []
datafolder = []
nel = []
depth = []
npos = []
end = []
zenith = [15,30,45]
energy = [16.5,17.4,17.8]
eventno = ['000010','000014','000018','000025','000022','000015']
fileno = [] 
#####################################same energy diff zenith ####################################################################

datafolder.append(sorted(glob.glob("/vol/astro7/lofar/krishna/ret/000010/*/*.long")))
datafolder.append(sorted(glob.glob("/vol/astro7/lofar/krishna/ret/000014/*/*.long")))
datafolder.append(sorted(glob.glob("/vol/astro7/lofar/krishna/ret/000018/*/*.long")))

################################## diff energy same zenith #######################################################################

datafolder.append(sorted(glob.glob("/vol/astro7/lofar/krishna/ret/000025/*/*.long")))
datafolder.append(sorted(glob.glob("/vol/astro7/lofar/krishna/ret/000022/*/*.long")))
datafolder.append(sorted(glob.glob("/vol/astro7/lofar/krishna/ret/000015/*/*.long")))

###########************************************************************************************************########################
#print(datafolder[2])

for index in range(0,6):
	xmaxs.append([])
	xmax_fileorder.append([])
	for longfile in datafolder[index]:
		#print(longfile)
		#if os.stat(longfile).st_size == 0:
		#	os.remove(longfile)
		#else:
		hillas = np.genfromtxt(re.findall("PARAMETERS.*",open(longfile,'r').read()))[2:] 
		xmaxs[index].append(hillas.T[2])
		#print(longfile," ",hillas.T[2])
		xmax_fileorder[index].append(hillas.T[2])
	avg.append(findMiddle(xmaxs[index]))
	for k in range(len(xmaxs[index])):
		if xmax_fileorder[index][k] == avg[index]: fileno.append(k)
#print(xmaxs,"\n")
print(avg)

for index in range(0,3):
	print(datafolder[index][fileno[index]])
	shutil.copy(datafolder[index][fileno[index]], "/vol/astro7/lofar/krishna/ret/long_prof_files_2/energy17.4/zenith{0}.long".format(zenith[index]) )
for index in range(3,6):
	print(datafolder[index][fileno[index]])
	shutil.copy(datafolder[index][fileno[index]], "/vol/astro7/lofar/krishna/ret/long_prof_files_2/zenith45/energy{0}.long".format(energy[index-3]) )


for index in range(0,6):	
	long_shower.makeshower(avg[index],eventno[index],index)


i=0

file1 = sorted(glob.glob("/vol/astro7/lofar/krishna/ret/long_prof_files_2/energy17.4/*.long"))
file2 = sorted(glob.glob("/vol/astro7/lofar/krishna/ret/long_prof_files_2/zenith45/*.long"))
for files in file1:
	with open(files, 'rt') as f:
		nel.append([])
		depth.append([])
		npos.append([])
		print(f)
		lines = f.readlines()[2:]
		for x in lines:
			d = x.split()[0]
			if d == 'LONGITUDINAL': break
			nel[i].append(x.split()[3])
			depth[i].append(x.split()[0])
			npos[i].append(x.split()[2])
		depth[i] = [float(x) for x in depth[i]]
		nel[i] = [(float(x)) for x in nel[i]]
		npos[i] = [(float(x)) for x in npos[i]]
		nel[i]= [x+y for x,y in zip(nel[i],npos[i])]
		end.append(depth[i][len(depth[i])-1])
		depth[i]=depth[i][:-1]
		nel[i]=nel[i][:-1]
		i=i+1

C = ['k','y','r'] 
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111)
for i in range(0,len(nel)): 
	ax.plot(nel[i],depth[i],'-*',markersize=4,linewidth=1,color=C[i], label = 'zenith{0},xmax={1}'.format(zenith[i],avg[i]),alpha = 0.6 )
	ax.axhline(y=end[i],ls = 'dashed',color=C[i])
	ax.annotate('{0}'.format(end[i]), (4.0,end[i]), color=C[i])
plt.xlabel("log10_Ne-e+")
plt.ylabel("depth(g/cm2)")
	#ax.annotate("{0}".format(end[i]), (0, end[i]))
ax.invert_yaxis()
plt.legend()
plt.savefig("/vol/astro7/lofar/krishna/ret/long_prof_files_2/energy17.4/longprof.png")
plt.close()

#*****************************************************************************************************************************************

nel.clear()
depth.clear()
npos.clear()
end.clear()
i=0 
for files in file2:
	with open(files, 'rt') as f:
		nel.append([])
		depth.append([])
		npos.append([])
		print(f)
		lines = f.readlines()[2:]
		for x in lines:
			d = x.split()[0]
			if d == 'LONGITUDINAL': break
			nel[i].append(x.split()[3])
			depth[i].append(x.split()[0])
			npos[i].append(x.split()[2])
		depth[i] = [float(x) for x in depth[i]]
		nel[i] = [(float(x)) for x in nel[i]]
		npos[i] = [(float(x)) for x in npos[i]]
		nel[i]= [x+y for x,y in zip(nel[i],npos[i])]
		end.append(depth[i][len(depth[i])-1])
		depth[i]=depth[i][:-1]
		nel[i]=nel[i][:-1]
		i=i+1
zenith = [15,30,45]
energy = [16.5,17.4,17.8]
C = ['k','y','r'] 
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111)

for i in range(0,len(nel)): 
	print(avg[i+3])
	ax.plot(nel[i],depth[i],'-*',markersize=4,linewidth=1,color=C[i], label = 'energy{0}, xmax = {1}'.format(energy[i],avg[i+3]),alpha = 0.6 )
	ax.axhline(y=end[i],ls = 'dashed',color=C[i])
	ax.annotate('{0}'.format(end[i]), (4.0,end[i]), color=C[i])
plt.xlabel("log10_Ne-e+")
plt.ylabel("depth(g/cm2)")
	#ax.annotate("{0}".format(end[i]), (0, end[i]))
ax.invert_yaxis()
plt.legend()
plt.savefig("/vol/astro7/lofar/krishna/ret/long_prof_files_2/zenith45/longprof.png")

