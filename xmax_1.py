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
from matplotlib.backends.backend_pdf import PdfPages
################################## location_of_stations ############################################################################

array_file='array_5a_1.txt'
positions=np.genfromtxt(array_file,skip_header=1)


################################## directories ###################################################################################

eventno = '000011'
#sim_director='/vol/astro7/lofar/krishna/cross_cal/reconstruction/data/'
inputdir='/vol/astro7/lofar/krishna/ret/filt_files/'
simfile_50_350 = os.path.join(inputdir, 'SIM{0}_{1}_{2}.filt'.format(eventno,50,350))
simfile_30_80 = os.path.join(inputdir, 'SIM{0}_{1}_{2}.filt'.format(eventno,30,80))

print('simfile 50-350: {0}'.format(simfile_50_350))
print('simfile 30-80: {0}'.format(simfile_30_80))

################################## open simfiles ###################################################################################

with open(simfile_30_80, 'rb') as pickle_file : 
	siminfo_30_80 = pickle.load(pickle_file,encoding='latin1')

with open(simfile_50_350, 'rb') as pickle_file:
    siminfo_50_350 = pickle.load(pickle_file,encoding='latin1')

primary_type=siminfo_50_350['primary']

nsimprot=np.sum(primary_type==14)   # number of proton simulations
nsimiron=np.sum(primary_type>14)    # number of iron simulations
l=nsimprot+nsimiron

print('number of proton simulations: {0}'.format(nsimprot))
print('number of iron simulations: {0}'.format(nsimiron))
print('total number of simulations : {0}'.format(l))

################################# exclude_those_that_has_extreme_xmax ##################################################################

xmax_values=siminfo_50_350['hillas'].T[2]
avg=np.average(xmax_values)
print(xmax_values)
idx = (np.abs(xmax_values - avg)).argmin()

ind_high=heapq.nlargest(2, range(len(xmax_values)), key=xmax_values.__getitem__)
ind_low=heapq.nsmallest(2, range(len(xmax_values)), key=xmax_values.__getitem__)   #why high and low????????
ind_outside=np.where(np.abs(xmax_values-avg)>120)
no_fit=np.concatenate((ind_high,ind_low,ind_outside), axis=None)
pp = PdfPages('/vol/astro7/lofar/krishna/ret/results/event{0}.pdf'.format(eventno))
#low = open("/vol/astro7/lofar/krishna/ret/results/{0}_30_80.txt".format(eventno),"w")
high = open("/vol/astro7/lofar/krishna/ret/results/{0}_50_350.txt".format(eventno),"w")



################################# simulate_each data ####################################################################################

for fit_n in range(0,l):
	if fit_n not in no_fit:
		print("********************",fit_n,"***************************************************")
		print('fitting for xmax: ',xmax_values[fit_n])
		core_x,core_y=fit_helper.get_core(50)
		print('core position: {0:.2f}, {1:.2f}'.format(core_x,core_y))
		
		niterations=3
		chi_mult=5
		flu_cut=1.0
		relerror = 0.10 #####to add error to data pretending as event 
			
		print('*****Fitting 50-350 MHz*****')
		fit_info_50_350=fit_helper.do_fit(fit_n,core_x,core_y,positions,siminfo_50_350,relerror,niterations,chi_mult,flu_cut)
		print('*****Fitting 30-80 MHz*****')
		fit_info_30_80=fit_helper.do_fit(fit_n,core_x,core_y,positions,siminfo_30_80,relerror,niterations,chi_mult,flu_cut)
		
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

		bestsim_30_80=fit_info_30_80['bestsim']
		antenna_fluence_30_80=fit_info_30_80['antenna_fluence']
		energy_scale_30_80=fit_info_30_80['energy_scale']
		x_err_30_80=fit_info_30_80['x_err']
		y_err_30_80=fit_info_30_80['y_err']
		xmax_reco_30_80=fit_info_30_80['xmax_reco']
		sim_tot_power_30_80=fit_info_30_80['sim_tot_power']
		rbf_30_80=fit_info_30_80['rbf']
		radiochi2_30_80=fit_info_30_80['radiochi2']
		parabola_30_80=fit_info_30_80['parabola']
		xmax_fit_30_80=fit_info_30_80['fit_x']
		reduced_chi2_fit_30_80=fit_info_30_80['fit_y']
		ndf_radio_30_80=fit_info_30_80['ndf_radio']
		fitconverged_30_80=fit_info_30_80['fitconverged']	
		nsimant=len(sim_antenna_position)	
		print('x,y err 30-80: {0:.2f}, {1:.2f}'.format(x_err_30_80,y_err_30_80))
		print('x,y err 50-350: {0:.2f}, {1:.2f}'.format(x_err_50_350,y_err_50_350))	
		
		print(antenna_fluence_30_80)
		print(antenna_fluence_50_350)
		delxmax_30_80 = abs(xmax_reco_30_80-xmax_true)
		delxmax_50_350 = abs(xmax_reco_50_350-xmax_true)
		################################## PLOTTING ###################################################
		if fitconverged_30_80 == True and fitconverged_50_350 == True : 
			#low.write("%d\t%s\t%s\t%s\t%.3f\t%s\t%s\t%.3f\t%d \t %.3f \t %.3f\t%.3f\t %.3f \n" % (fit_n,energy,zenith,azimuth,np.sqrt(energy_scale_30_80),xmax_true,xmax_reco_30_80, delxmax_30_80 ,fitconverged_30_80,core_x,core_y,x_err_30_80,y_err_30_80))
			high.write("%d\t%s\t%s\t%s\t%.3f\t%s\t%s\t%.3f\t%d \t %.3f \t %.3f\t%.3f\t %.3f \n" % (fit_n,energy,zenith,azimuth,np.sqrt(energy_scale_50_350),xmax_true,xmax_reco_50_350, delxmax_50_350,fitconverged_50_350,core_x,core_y,x_err_50_350,y_err_50_350))
			fig = plt.figure(figsize=(7,7))
			#ax1 = fig.add_subplot(2,3,1,aspect=1)
			#ax2 = fig.add_subplot(2,2,1,aspect=1)
			#ax3 = fig.add_subplot(2,2,2)
			#ax4 = fig.add_subplot(2,3,4,aspect=1)
			ax5 = fig.add_subplot(2,2,3,aspect=1)
			ax6 = fig.add_subplot(2,2,4)
			    
			fig.suptitle('Event {0}: energy = {1:.2e} zenith = {2:.2f} simeventno : {3}'.format(eventno,1e9*energy,zenith*180/np.pi,fit_n)) 

			#*********************** 30-80 ***********************
			dist_scale=1.5*np.max(np.sqrt((positions[:,0]-core_x-x_err_30_80)*(positions[:,0]-core_x-x_err_30_80)+(positions[:,1]-core_y-y_err_30_80)*(positions[:,1]-core_y-y_err_30_80)))
			ti = np.linspace(-dist_scale, dist_scale, 150)
			XI, YI = np.meshgrid(ti, ti)
			selection_30_80 = np.unique(np.array(np.isfinite(sim_tot_power_30_80))*np.arange(nsimant))
			rbf_ground_30_80 = intp.Rbf(sim_antenna_position.T[0], sim_antenna_position.T[1], sim_tot_power_30_80, smooth =0, function='quintic')

			ZI_30_80 = rbf_ground_30_80(XI, YI)*energy_scale_30_80
			print(np.min(ZI_30_80))
			dist_scale=1.5*np.max(np.sqrt((positions[:,0]-core_x-x_err_50_350)*(positions[:,0]-core_x-x_err_50_350)+(positions[:,1]-core_y-y_err_50_350)*(positions[:,1]-core_y-y_err_50_350)))
			ti = np.linspace(-dist_scale, dist_scale, 150)
			XI, YI = np.meshgrid(ti, ti)
			selection_50_350 = np.unique(np.array(np.isfinite(sim_tot_power_50_350))*np.arange(nsimant))
			rbf_ground_50_350 = intp.Rbf(sim_antenna_position.T[0], sim_antenna_position.T[1], sim_tot_power_50_350, smooth =0, function='quintic')

			ZI_50_350 = rbf_ground_50_350(XI, YI)*energy_scale_50_350
			maxp = np.max([np.max(antenna_fluence_30_80),np.max(ZI_30_80),np.max(antenna_fluence_50_350),np.max(ZI_50_350)])
			maxp_30_80 = np.max([np.max(antenna_fluence_30_80),np.max(ZI_30_80)])

			#maxp = np.max([np.max(antenna_fluence_30_80),np.max(ZI_30_80)])
			'''
			ax1.pcolor(XI+core_x+x_err_30_80, YI+core_y+y_err_30_80, ZI_30_80,vmax=maxp, vmin=0,cmap=cm.jet)
			ax1.scatter(positions[:,0],positions[:,1],50,c=antenna_fluence_30_80,vmax=maxp, vmin=0,cmap=cm.jet,edgecolors='white')
			ax1.scatter(sim_antenna_position[:,0]+core_x+x_err_30_80,sim_antenna_position[:,1]+core_y+y_err_30_80,10, c=sim_tot_power_30_80*energy_scale_30_80,vmax=maxp, vmin=0,cmap=cm.jet)
			ax1.scatter(core_x,core_y,marker='+')
			ax1.scatter(core_x+x_err_30_80,core_y+y_err_30_80,marker='+')
			ax1.set_xlim((-dist_scale+core_x+x_err_30_80,dist_scale+core_x+x_err_30_80))
			ax1.set_ylim((-dist_scale+core_y+y_err_30_80,dist_scale+core_y+y_err_30_80))
			ax1.set_title("Footprint- ground plane",fontsize=12)
			ax1.set_xlabel("West-East (m)")
			ax1.set_ylabel("South-North (m)")
			'''
			axdist_ant = np.sqrt(pos_ant_UVW[:,0]*pos_ant_UVW[:,0]+pos_ant_UVW[:,1]*pos_ant_UVW[:,1])
			dist_scale = 1.2*np.max(axdist_ant)

			ti = np.linspace(-dist_scale, dist_scale, 150)
			XI, YI = np.meshgrid(ti, ti)
			ZI_power_showerplane_30_80 = rbf_30_80(XI, YI) * energy_scale_30_80
			'''
			ax2.pcolor(XI, YI, ZI_power_showerplane_30_80, vmax=maxp_30_80, vmin=0,cmap=cm.jet, linewidth=0, rasterized=True)
			ax2.scatter(pos_sim_UVW[:,0],pos_sim_UVW[:,1],10,sim_tot_power_30_80*energy_scale_30_80, vmax=maxp_30_80, vmin=0,cmap=cm.jet)
			ax2.scatter(pos_ant_UVW[:,0],pos_ant_UVW[:,1],50,antenna_fluence_30_80, vmax=maxp_30_80, vmin=0,cmap=cm.jet,edgecolors='white')
			ax2.scatter(0,0,marker='+')
			ax2.set_xlim((-dist_scale,dist_scale))
			ax2.set_ylim((-dist_scale,dist_scale))
			ax2.set_title("Footprint- shower plane 30-80MHz",fontsize=12)
			ax2.set_xlabel("vxB (m)")
			ax2.set_ylabel("vx(vxB) (m)")
			ax2.text(-0.9*dist_scale, 0.8*dist_scale, 'Xmax = {0:.1f} g/cm2'.format(abs(xmax_reco_30_80-xmax_true)), color='white', weight="bold",fontsize=10)
			ax2.text(-0.9*dist_scale, 0.6*dist_scale,  'E = {0:.3f} $\%$' .format(np.sqrt(energy_scale_30_80)), color='white',weight="bold", fontsize=10)


			if fitconverged_30_80==True:
			    y_range=radiochi2_30_80[radiochi2_30_80<500]
			    ax3.plot(hillas[2],radiochi2_30_80,'.')
			    ax3.plot(xmax_fit_30_80,reduced_chi2_fit_30_80,'x',color='red')
			    t = np.linspace(xmax_true+200, xmax_true-200, 100)
			    ax3.plot(t,parabola_30_80(t)/ndf_radio_30_80,'-',color='m', lw=1)
			    ax3.axvline(x=xmax_true,color='grey')
			    ax3.axvline(x=xmax_reco_30_80,color='black',linestyle=':')

			    ax3.set_ylim(-2,np.max(y_range)*1.2)
			    ax3.set_xlabel("Xmax (g/cm2)")
			    ax3.set_ylabel("Chi2")
			'''
			#*********************** 30-80 ***********************


			'''
			ax4.pcolor(XI+core_x+x_err_50_350, YI+core_y+y_err_50_350, ZI_50_350,vmax=maxp, vmin=0,cmap=cm.jet)
			ax4.scatter(positions[:,0],positions[:,1],50,c=antenna_fluence_50_350,vmax=maxp, vmin=0,cmap=cm.jet,edgecolors='white')
			ax4.scatter(sim_antenna_position[:,0]+core_x+x_err_50_350,sim_antenna_position[:,1]+core_y+y_err_50_350,10, c=sim_tot_power_50_350*energy_scale_50_350,vmax=maxp, vmin=0,cmap=cm.jet)
			ax4.scatter(core_x,core_y,marker='+')
			ax4.scatter(core_x+x_err_50_350,core_y+y_err_50_350,marker='+')
			ax4.set_xlim((-dist_scale+core_x+x_err_50_350,dist_scale+core_x+x_err_50_350))
			ax4.set_ylim((-dist_scale+core_y+y_err_50_350,dist_scale+core_y+y_err_50_350))
			ax4.set_title("Footprint- ground plane",fontsize=12)
			ax4.set_xlabel("West-East (m)")
			ax4.set_ylabel("South-North (m)")
			'''
			axdist_ant = np.sqrt(pos_ant_UVW[:,0]*pos_ant_UVW[:,0]+pos_ant_UVW[:,1]*pos_ant_UVW[:,1])
			dist_scale = 1.2*np.max(axdist_ant)

			ti = np.linspace(-dist_scale, dist_scale, 150)
			XI, YI = np.meshgrid(ti, ti)
			ZI_power_showerplane_50_350 = rbf_50_350(XI, YI) * energy_scale_50_350
			pcm = ax5.pcolor(XI, YI, ZI_power_showerplane_50_350, vmax=maxp, vmin=0,cmap=cm.jet, linewidth=0, rasterized=True)
			plt.colorbar(pcm)
			ax5.scatter(pos_sim_UVW[:,0],pos_sim_UVW[:,1],10,sim_tot_power_50_350*energy_scale_50_350, vmax=maxp, vmin=0,cmap=cm.jet)
			ax5.scatter(pos_ant_UVW[:,0],pos_ant_UVW[:,1],50,antenna_fluence_50_350, vmax=maxp, vmin=0,cmap=cm.jet,edgecolors='white')
			ax5.scatter(0,0,marker='+')
			ax5.set_xlim((-dist_scale,dist_scale))
			ax5.set_ylim((-dist_scale,dist_scale))
			ax5.set_title("Footprint- shower plane 50-350MHz",fontsize=12)
			ax5.set_xlabel("vxB (m)")
			ax5.set_ylabel("vx(vxB) (m)")
			ax5.text(-0.9*dist_scale, 0.8*dist_scale, 'Xmax = {0:.1f} g/cm2'.format(abs(xmax_reco_50_350-xmax_true)), color='white', weight="bold",fontsize=10)
			ax5.text(-0.9*dist_scale, 0.6*dist_scale,  'E = {0:.3f} $\%$' .format(np.sqrt(energy_scale_50_350)), color='white',weight="bold", fontsize=10)


			if fitconverged_50_350==True:

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

			#plt.tight_layout()
			fig.tight_layout(rect=[0, 0.03, 1, 0.95])
			#plt.savefig("/vol/astro7/lofar/krishna/ret/scripts/pics/gotfig{0}".format(fit_n))
			pp.savefig()
pp.close()
low.close()
high.close()				
				



