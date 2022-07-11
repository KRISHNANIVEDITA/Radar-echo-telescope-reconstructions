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
import fit_helper
from matplotlib import cm
import imp  
from scipy import stats
import heapq
import math
from matplotlib.backends.backend_pdf import PdfPages
################################## location_of_stations ############################################################################
zen = [15,30,45,45,45,45]
en = [17.4,17.4,17.4,16.5,17.4,17.8]
def makeshower(avg,eventno,index):
	array_file='array_5a.txt'
	positions=np.genfromtxt(array_file,skip_header=1)


	################################## directories ###################################################################################

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

	#**********************************************************************************************************************************
	xmax_values=siminfo_50_350['hillas'].T[2]
	for i in range(len(xmax_values)):
		if xmax_values[i] == avg:
			fit_n = i
	print(fit_n)
	print("********************",fit_n,"***************************************************")
	print('fitting for xmax: ',xmax_values[fit_n])

	core_x,core_y=fit_helper.get_core(160)
	print('core position: {0:.2f}, {1:.2f}'.format(core_x,core_y))
	niterations=3
	chi_mult=5
	flu_cut=1.0
	relerror = 0.10 #####to add error to data pretending as event 
		
	#print('*****Fitting 50-350 MHz*****')
	fit_info_50_350=fit_helper.do_fit(fit_n,core_x,core_y,positions,siminfo_50_350,relerror,niterations,chi_mult,flu_cut)
	
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
	if fitconverged_50_350 == True:
		fig = plt.figure(figsize=(7,7))
		ax5 = fig.add_subplot(1,1,1,aspect=1)
		#ax6 = fig.add_subplot(2,1,1,aspect =1)
		fig.suptitle('Event {0}: energy = 10^{1:.2f} zenith = {2:.2f} '.format(eventno,math.log10(energy*10**9),zenith*180/np.pi)) 
		dist_scale=1.5*np.max(np.sqrt((positions[:,0]-core_x-x_err_50_350)*(positions[:,0]-core_x-x_err_50_350)+(positions[:,1]-core_y-y_err_50_350)*(positions[:,1]-core_y-y_err_50_350)))
		ti = np.linspace(-dist_scale, dist_scale, 150)
		XI, YI = np.meshgrid(ti, ti)
		selection_50_350 = np.unique(np.array(np.isfinite(sim_tot_power_50_350))*np.arange(nsimant))
		#rbf_ground_50_350 = intp.Rbf(sim_antenna_position.T[0], sim_antenna_position.T[1], sim_tot_power_50_350, smooth =0, function='quintic')
		#ZI_50_350 = rbf_ground_50_350(XI, YI)*energy_scale_50_350

		axdist_ant = np.sqrt(pos_ant_UVW[:,0]*pos_ant_UVW[:,0]+pos_ant_UVW[:,1]*pos_ant_UVW[:,1])
		dist_scale = 1.2*np.max(axdist_ant)
		ti = np.linspace(-dist_scale, dist_scale, 150)
		XI, YI = np.meshgrid(ti, ti)
		ZI_power_showerplane_50_350 = rbf_50_350(XI, YI) * energy_scale_50_350
		maxp = max(np.max(antenna_fluence_50_350),np.max(ZI_power_showerplane_50_350))
		print(maxp)
		ax5.pcolor(XI, YI, ZI_power_showerplane_50_350, vmax=maxp, vmin=0,cmap=cm.jet, linewidth=0, rasterized=True)
		sc = ax5.scatter(pos_sim_UVW[:,0],pos_sim_UVW[:,1],10,sim_tot_power_50_350*energy_scale_50_350, vmax=maxp, vmin=0,cmap=cm.jet)
		ax5.scatter(pos_ant_UVW[:,0],pos_ant_UVW[:,1],50,antenna_fluence_50_350, vmax=maxp, vmin=0,cmap=cm.jet,edgecolors='white')
		plt.colorbar(sc)
		ax5.scatter(0,0,marker='+')
		ax5.set_xlim((-dist_scale,dist_scale))
		ax5.set_ylim((-dist_scale,dist_scale))
		ax5.set_title("Footprint- shower plane 50-350MHz",fontsize=12)
		ax5.set_xlabel("vxB (m)")
		ax5.set_ylabel("vx(vxB) (m)")
		ax5.text(-0.9*dist_scale, 0.8*dist_scale, 'Xmax_true = {0:.1f} g/cm2'.format(abs(xmax_true)), color='white', weight="bold",fontsize=10)
		ax5.text(-0.9*dist_scale, 0.6*dist_scale, 'Xmax_reco = {0:.1f} g/cm2'.format(xmax_reco_50_350), color='white', weight="bold",fontsize=10)
		ax5.text(-0.9*dist_scale, 0.4*dist_scale,  'E = {0:.3f} $\%$' .format(np.sqrt(energy_scale_50_350)), color='white',weight="bold", fontsize=10)
		fig.tight_layout(rect=[0, 0.03, 1, 0.95])
		if index < 3 :
			plt.savefig("/vol/astro7/lofar/krishna/ret/long_prof_files_2/energy17.4/shower{0}_{1}_{2}.png".format(index,en[index],zen[index]))
		else:  
			plt.savefig("/vol/astro7/lofar/krishna/ret/long_prof_files_2/zenith45/shower{0}_{1}_{2}.png".format(index-3,en[index],zen[index]))
		
		fig = plt.figure(figsize=(7,7))
		ax6 = fig.add_subplot(2,1,1)
		y_range=radiochi2_50_350[radiochi2_50_350<500]
		ax6.plot(hillas[2],radiochi2_50_350,'.')
		ax6.plot(xmax_fit_50_350,reduced_chi2_fit_50_350,'x',color='red')
		t = np.linspace(xmax_true+200, xmax_true-200, 100)
		ax6.plot(t,parabola_50_350(t)/ndf_radio_50_350,'-',color='m', lw=1)
		ax6.axvline(x=xmax_true,color='grey')
		ax6.axvline(x=xmax_reco_50_350,color='black',linestyle=':')
		ax6.set_ylim(-2,np.max(y_range)*1.2)
		ax6.set_xlabel("Xmax (g/cm2)")
		ax6.set_ylabel("Chi2")
		fig.tight_layout(rect=[0, 0.03, 1, 0.95])
		if index < 3:
			plt.savefig("/vol/astro7/lofar/krishna/ret/long_prof_files_2/energy17.4/fitting{0}_{1}_{2}.png".format(index,en[index],zen[index]))
		else:
			plt.savefig("/vol/astro7/lofar/krishna/ret/long_prof_files_2/zenith45/fitting{0}_{1}_{2}.png".format(index-3,en[index],zen[index]))

	else:
		print("sorry")		
		

		
		
		
