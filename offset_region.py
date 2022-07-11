import numpy as np
import glob
import pickle
from optparse import OptionParser
import os
import scipy.interpolate as intp
import scipy.optimize as opt
from scipy.special import gamma
import random
import matplotlib.pyplot as plt
import fit_offset
from matplotlib import cm
from scipy import stats
import heapq
from matplotlib.backends.backend_pdf import PdfPages
################################## ###########################################################################

#####################################location_of_stations #######################################################################################3
array_file='array_5a_1.txt'
positions=np.genfromtxt(array_file,skip_header=1)


################################## directories ###################################################################################
parser = OptionParser()
parser.add_option('-e', '--eventnumber', default ='000026', help = 'event number')
(options, args) = parser.parse_args()
eventno=str(options.eventnumber)
#sim_director='/vol/astro7/lofar/krishna/cross_cal/reconstruction/data/'
inputdir='/vol/astro7/lofar/krishna/ret/filt_files/'
simfile_50_350 = os.path.join(inputdir, 'SIM{0}_{1}_{2}.filt'.format(eventno,50,350))

print('simfile 50-350: {0}'.format(simfile_50_350))


################################## open simfiles ###################################################################################


with open(simfile_50_350, 'rb') as pickle_file:
    siminfo_50_350 = pickle.load(pickle_file,encoding='latin1')

primary_type=siminfo_50_350['primary']

nsimprot=np.sum(primary_type==14)   # number of proton simulations
nsimiron=np.sum(primary_type>14)    # number of iron simulations
l=nsimprot+nsimiron
print('number of proton simulations: {0}'.format(nsimprot))
print('number of iron simulations: {0}'.format(nsimiron))
print('total number of simulations : {0}'.format(l))

################## exclude_those_that_has_extreme_xmax ##################################################################

xmax_values=siminfo_50_350['hillas'].T[2]
xmax_values = [float(x) for x in xmax_values]
avg=np.average(xmax_values[1:])
print(xmax_values)
idx = (np.abs(xmax_values - avg)).argmin()

ind_high=heapq.nlargest(2, range(len(xmax_values)), key=xmax_values.__getitem__)
ind_low=heapq.nsmallest(2, range(len(xmax_values)), key=xmax_values.__getitem__)   #why high and low????????
ind_outside=np.where(np.abs(xmax_values-avg)>120)
no_fit=np.concatenate((ind_high,ind_low,ind_outside), axis=None)
#pp = PdfPages('/vol/astro7/lofar/krishna/ret/results/event{0}.pdf'.format(eventno))
high = open("/vol/astro7/lofar/krishna/ret/data/{0}.txt".format(eventno),"w")
fluence = open("/vol/astro7/lofar/krishna/ret/fluence/{0}_fluence.txt".format(eventno),"w")

print(no_fit)
################################# simulate_each data ####################################################################################
for fit_n in range(0,l):	
	if fit_n not in no_fit:
		print('fitting for xmax: ',xmax_values[fit_n])
		for k in range(0,200):
			core_x,core_y=fit_offset.get_core(200)
			print('core position: {0:.2f}, {1:.2f}'.format(core_x,core_y))
			
			niterations=3
			chi_mult=5
			flu_cut=1.0
			relerror = 0.10 #####to add error to data pretending as event 
				
			#print('*****Fitting 50-350 MHz*****')
			fit_info_50_350=fit_offset.do_fit(fit_n,core_x,core_y,positions,siminfo_50_350,relerror,niterations,chi_mult,flu_cut)
			
			energy=fit_info_50_350['energy']              
			azimuth=fit_info_50_350['azimuth']
			zenith=fit_info_50_350['zenith']
			xmax_true=fit_info_50_350['xmax_true']
			sim_antenna_position=fit_info_50_350['sim_antenna_position']
			pos_ant_UVW=fit_info_50_350['pos_ant_UVW']
			pos_sim_UVW=fit_info_50_350['pos_sim_UVW']
			sim_antenna_position=fit_info_50_350['sim_antenna_position']
			hillas=fit_info_50_350['hillas']


			bestsim_50_350=fit_info_50_350['bestsim']
			antenna_fluence_50_350=fit_info_50_350['antenna_fluence']
			energy_scale_50_350=fit_info_50_350['energy_scale']
			x_err_50_350=fit_info_50_350['x_err']
			y_err_50_350=fit_info_50_350['y_err']
			xmax_reco_50_350=fit_info_50_350['xmax_reco']
			sim_tot_power_50_350=fit_info_50_350['sim_tot_power']
			rbf_50_350=fit_info_50_350['rbf']
			radiochi2_50_350=fit_info_50_350['radiochi2']
			parabola_50_350=fit_info_50_350['parabola']
			xmax_fit_50_350=fit_info_50_350['fit_x']
			reduced_chi2_fit_50_350=fit_info_50_350['fit_y']
			ndf_radio_50_350=fit_info_50_350['ndf_radio']
			fitconverged_50_350=fit_info_50_350['fitconverged']
			id_var_50_350 = fit_info_50_350['id_var']

			#print("idvar",id_var_50_350)
			nsimant=len(sim_antenna_position)	
			#print('x,y err 50-350: {0:.2f}, {1:.2f}'.format(x_err_50_350,y_err_50_350))	
			
			#print(antenna_fluence_50_350)
			delxmax_50_350 = abs(xmax_reco_50_350-xmax_true)
			
			################################## PLOTTING ###################################################
			high.write("%d\t%s\t%s\t%s\t%.3f\t%s\t%s\t%.3f\t%d \t %.3f \t %.3f\t%.3f\t %.3f\t%d \n" % (fit_n,energy,zenith,azimuth,np.sqrt(energy_scale_50_350),xmax_true,xmax_reco_50_350, delxmax_50_350,fitconverged_50_350,core_x,core_y,x_err_50_350,y_err_50_350,id_var_50_350))
			fluence.write("%s\n" % ( " ".join( repr(e) for e in antenna_fluence_50_350)))
high.close()
fluence.close()
