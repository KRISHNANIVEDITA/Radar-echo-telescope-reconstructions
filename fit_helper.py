import numpy as np
import glob
import pickle
from optparse import OptionParser
import os
import scipy.interpolate as intp
import scipy.optimize as opt
from scipy.special import gamma
# import sympy 
from sympy import * 
import random
def GetUVW(pos, cx, cy, zen, az):
	relpos = pos-np.array([cx,cy,2400])
	inc=-84.8/180.*np.pi
	B = np.array([0,np.cos(inc),-np.sin(inc)])
	v = np.array([-np.cos(az)*np.sin(zen),-np.sin(az)*np.sin(zen),-np.cos(zen)])
	#print v
	vxB = np.array([v[1]*B[2]-v[2]*B[1],v[2]*B[0]-v[0]*B[2],v[0]*B[1]-v[1]*B[0]])
	vxB = vxB/np.linalg.norm(vxB)
	vxvxB = np.array([v[1]*vxB[2]-v[2]*vxB[1],v[2]*vxB[0]-v[0]*vxB[2],v[0]*vxB[1]-v[1]*vxB[0]])
	return np.array([np.inner(vxB,relpos),np.inner(vxvxB,relpos),np.inner(v,relpos)]).T

def do_fit(simevent,core_x,core_y,positions,siminfo,relerror,niterations,chi_mult,flu_cut):
	zenith=siminfo['zenith']
	azimuth=siminfo['azimuth']
	energy=siminfo['energy']
	hillas=siminfo['hillas'].T
	longprofile=siminfo['longprofile']
	Xground=siminfo['Xground']
	sim_antenna_position=siminfo['antenna_position']
	sim_power=siminfo['fluence']
	primary_type=siminfo['primary']
	nant=len(positions)
	ant_sel=np.ones([nant],dtype=bool)  # saying all antennas will be used
	ant_cut=np.ones([nant],dtype=bool)  # saying all antennas will be used
	nsel_ant = np.sum(ant_sel)
	data_zenith=zenith[0]
	data_azimuth=azimuth[0]
	sim_tot_power=sim_power#np.sum(sim_power,axis=2)
	nsim=sim_tot_power.shape[0]
	nsimant=sim_tot_power.shape[1]
	nsim_prot=np.sum(primary_type==14)
	pos_sim_UVW = np.zeros([nsim,nsimant,3])  # antenna positions in the shower plane 
	rbf= np.ndarray([nsim],dtype=object)  # to store interpolation funtion
	#print(sim_power)

	# interpolating power from simulations
	for i in np.arange(nsim):
		pos_sim_UVW[i,:,:] = GetUVW(sim_antenna_position[i,:,:],0,0, zenith[0], azimuth[0])
		selection=np.array(np.isfinite(sim_power[i,:]))*np.array(np.isfinite(sim_power[i,:]))
		rbf[i] = intp.Rbf(pos_sim_UVW[i,selection,0], pos_sim_UVW[i,selection,1], sim_tot_power[i,selection],smooth =0,function='quintic')

	def radio_fitfunc(f,dpos,n,cx,cy,az,zen):
		pos_ant_UVW = GetUVW(dpos, cx, cy, zen, az)
		interp_power = rbf[n](pos_ant_UVW[:,0],pos_ant_UVW[:,1])
		return f*interp_power
	radio_errfunc = lambda p,lofar_pow,lofar_err, lofar_pos,cx, cy, az, zen, n: (radio_fitfunc(p[0],lofar_pos,n,cx+p[1],cy+p[2],az,zen) - lofar_pow)/lofar_err
	
	radiochi2=np.zeros([nsim])
	p_ratio=np.zeros([nsim])
	xoffset=np.zeros([nsim])
	yoffset=np.zeros([nsim])
	simpower=np.zeros([nant,nsim])

	realxmax=hillas[2,simevent]
	pos_ant_UVW = GetUVW(positions, core_x, core_y, data_zenith, data_azimuth)
	dtotpower=rbf[simevent](pos_ant_UVW[:,0],pos_ant_UVW[:,1])
	dsigmatot=dtotpower*relerror
	# add gaussian "noise" to "data"
	#print('fluence: ', dtotpower)
	
	for i in np.arange(nant):
		dtotpower[i]=dtotpower[i] + random.gauss(0,dsigmatot[i])
		if dtotpower[i]<flu_cut:
			ant_cut[i]=0
	#print(ant_cut)
		
	def FitRoutine_radio_only(fit_args,iterations=niterations):
		paramset=np.ndarray([iterations,3]) # 3 free parameters in fit
		itchi2=np.zeros(iterations)
		for j in np.arange(iterations):
			initguess=[1, 200*np.random.rand()-100, 200*np.random.rand()-100] # Radio power scale factor, core X, core Y
			paramset[j], covar = opt.leastsq(radio_errfunc, initguess, args=fit_args)
			itchi2[j] = np.sum(np.square(radio_errfunc(paramset[j],*fit_args)))

		bestiter=np.argmin(itchi2)
		fitparam=paramset[bestiter]
		(xoffset, yoffset) = (fitparam[1], fitparam[2])
		chi2_rad=np.sum(np.square(radio_errfunc(fitparam, *fit_args)))
		return chi2_rad,fitparam[0],fitparam[1], fitparam[2]

	Desired_FitRoutine = FitRoutine_radio_only
	niterations=3
	id_var = -1
	if len(ant_cut[ant_cut==1])>=3:
		for i in np.arange(nsim): 
    			fit_args=(dtotpower[ant_sel],dsigmatot[ant_sel],positions[ant_sel],core_x,core_y,data_azimuth,data_zenith,i)
	    		radiochi2[i], p_ratio[i], xoffset[i], yoffset[i] = Desired_FitRoutine(fit_args,niterations)

		Desired_Chi2 = radiochi2
		radiochi2[simevent]=1000 
		bestsim=np.argmin(Desired_Chi2)
		xoff=xoffset[bestsim]
		yoff=yoffset[bestsim]
	
		fitparam=np.array([p_ratio[bestsim],xoffset[bestsim],yoffset[bestsim]])
		fit_args=(dtotpower,dsigmatot,positions,core_x,core_y,data_azimuth,data_zenith,bestsim)
		rad_rel_err= np.square(radio_errfunc([p_ratio[bestsim],xoffset[bestsim],yoffset[bestsim]],*fit_args))
		ant_sel=ant_sel*(rad_rel_err<16)
		#print("antenna selected less error: ",ant_sel)
		nsel_ant = np.sum(ant_sel)
		#print("Antennas flagged due error: ", nant-np.sum(ant_sel), " out of ", nant)
		ndf_radio = nsel_ant-3   #this is wrong- katie
		#print(nsel_ant)
		id_var = 0
		if len(ant_sel[ant_sel == 1])>=4:  
			for i in np.arange(nsim):
				fit_args=(dtotpower[ant_sel],dsigmatot[ant_sel],positions[ant_sel],core_x,core_y,data_azimuth,data_zenith,i)
				radiochi2[i],p_ratio[i],xoffset[i], yoffset[i] = Desired_FitRoutine(fit_args,niterations)	
			radiochi2[simevent]=1000
			bestsim=np.argmin(Desired_Chi2)
			xoff=xoffset[bestsim]
			yoff=yoffset[bestsim]

			for i in np.arange(nsim):
				simpower[:,i]=radio_fitfunc(p_ratio[i],positions,i,core_x+xoff,core_y+yoff,azimuth[0],zenith[0])

			pos_ant_UVW = GetUVW(positions, core_x+xoff, core_y+yoff, data_zenith, data_azimuth)
			axdist_ant = np.sqrt(pos_ant_UVW[:,0]*pos_ant_UVW[:,0]+pos_ant_UVW[:,1]*pos_ant_UVW[:,1])
	    
	    

			Desired_ndf = ndf_radio
			avg_chi2=np.average(Desired_Chi2[Desired_Chi2<500])
			urange=hillas[2,bestsim]+200
			drange=hillas[2,bestsim]-200
			chirange = Desired_Chi2[bestsim] + chi_mult*avg_chi2 * Desired_ndf
			fit_selection=np.zeros(nsim) # all points that have lower chi2 values on on side only
			for i in np.arange(nsim):
				if (np.sum(Desired_Chi2[(hillas[2,:]>hillas[2,i])] < Desired_Chi2[i])==0): fit_selection[i]=fit_selection[i]+1
				if (np.sum(Desired_Chi2[(hillas[2,:]<hillas[2,i])] < Desired_Chi2[i])==0): fit_selection[i]=fit_selection[i]+1
      
			fit_selection=fit_selection*(hillas[2,:]>drange)*(hillas[2,:]<urange)*((Desired_Chi2-1)<chirange)
			fit_selection[simevent]=0

			fo=2
			chi2fitparam_1=np.zeros([3])
			fitconverged_1=False
			#print('trying parabola fit')
			fit_y0=Desired_Chi2[(fit_selection>0)]
			fit_x0=hillas[2,(fit_selection>0)]
			if (np.sum(fit_selection)>fo):
				chi2fitparam_1, res_1, rank_1, singval_1, rcond_1 = np.polyfit(fit_x0,fit_y0,fo,full=True)
				pfit_1 = np.poly1d(chi2fitparam_1)
				r1=Derivative(hbola(x,[a,b]), x) 
				if (r1[(r1>drange)*(r1<urange)].size<3):
					if (r1[(r1>drange)*(r1<urange)].size>0): fitconverged_1=True
				xmaxreco=0
				fit_y_final=fit_y0
				fit_x_final=fit_x0
				n_sim_fit=len(hillas[2,(fit_selection>0)])
				res_1_min=1000
				id_var = 1
				fit_y0=Desired_Chi2[(fit_selection>0)]
				fit_x0=hillas[2,(fit_selection>0)]

				for n in np.arange(n_sim_fit):
					fit_x=np.delete(fit_x0,n)
					fit_y=np.delete(fit_y0,n)

				chi2fitparam_1, res_1, rank_1, singval_1, rcond_1 = np.polyfit(fit_x,fit_y,fo,full=True)
				pfit_1_temp = np.poly1d(chi2fitparam_1)
				r1_temp=pfit_1.deriv().r
				if (r1_temp[(r1_temp>drange)*(r1_temp<urange)].size<3):
					if (r1_temp[(r1_temp>drange)*(r1_temp<urange)].size>0): fitconverged_1=True
				xmaxreco=0

				if res_1<res_1_min:
					pfit_1=pfit_1_temp
					r1=r1_temp
					res_1_min=res_1
					fit_y_final=fit_y
					fit_x_final=fit_x
					

			else:
            			#print('Not! doing parabola fit')
            			fit_x_final=0
            			fit_y_final=0
            			pfit_1=0
            			id_var =2
			if (fitconverged_1):
				try:
					#print(r1)
					xmaxreco = r1[(r1>drange)*(r1<urange)][0]
				except:
					xmaxreco=0
					fitconverged_1=False
					#print('problem with r1')
			else:
				#print('didn\'t converge')
				pfit_1=0
				xmaxreco=0
			#print('Reconstructed Xmax = {0:.3f} g/cm2'.format(xmaxreco))
			#print('True Xmax ={0:.3f} g/cm2'.format(hillas.T[simevent][2]))
			#print('Energy reco ={0:.3f} '.format(np.sqrt(p_ratio[bestsim])))
			#print('Fit converged? {0} '.format(fitconverged_1))
			x_err=xoffset[bestsim]
			y_err=yoffset[bestsim]
			fit_info={'energy':energy[0],'azimuth':data_azimuth,'zenith':data_zenith,'hillas':hillas,'bestsim':bestsim,'antenna_fluence':dtotpower,'energy_scale':p_ratio[bestsim],'x_err':x_err,'y_err':y_err,'xmax_true':realxmax, 'xmax_reco':xmaxreco,'fitconverged':fitconverged_1,'sim_tot_power':sim_tot_power[bestsim],'sim_antenna_position':sim_antenna_position[bestsim],'pos_sim_UVW':pos_sim_UVW[bestsim],'pos_ant_UVW':pos_ant_UVW,'rbf':rbf[bestsim],'radiochi2':radiochi2,'parabola':pfit_1,'fit_x':fit_x_final, 'fit_y':fit_y_final,'ndf_radio':ndf_radio,'id_var':id_var}
	
     	
		else:
			#print("large error in antennas")
			pfit_1=0
			xmaxreco=0
			fit_x_final=0
			fit_y_final=0
			fitconverged_1=False
			bestsim=0
			p_ratio=1
			x_err=0
			y_err=0
			radiochi2=0
			fit_info={'energy':energy[0],'azimuth':data_azimuth,'zenith':data_zenith,'hillas':hillas,'bestsim':bestsim,'antenna_fluence':dtotpower,'energy_scale':1,'x_err':x_err,'y_err':y_err,'xmax_true':realxmax, 'xmax_reco':xmaxreco,'fitconverged':fitconverged_1,'sim_tot_power':sim_tot_power[simevent],'sim_antenna_position':sim_antenna_position[simevent],'pos_sim_UVW':pos_sim_UVW[simevent],'pos_ant_UVW':pos_ant_UVW,'rbf':rbf[simevent],'radiochi2':0,'parabola':0,'fit_x':0, 'fit_y':0,'ndf_radio':0,'id_var':id_var}

	else:
		#print('not enough antennas with signal')
		pfit_1=0
		xmaxreco=0
		fit_x_final=0
		fit_y_final=0
		fitconverged_1=False
		bestsim=0
		p_ratio=1
		x_err=0
		y_err=0
		radiochi2=0
		fit_info={'energy':energy[0],'azimuth':data_azimuth,'zenith':data_zenith,'hillas':hillas,'bestsim':bestsim,'antenna_fluence':dtotpower,'energy_scale':1,'x_err':x_err,'y_err':y_err,'xmax_true':realxmax, 'xmax_reco':xmaxreco,'fitconverged':fitconverged_1,'sim_tot_power':sim_tot_power[simevent],'sim_antenna_position':sim_antenna_position[simevent],'pos_sim_UVW':pos_sim_UVW[simevent],'pos_ant_UVW':pos_ant_UVW,'rbf':rbf[simevent],'radiochi2':0,'parabola':0,'fit_x':0, 'fit_y':0,'ndf_radio':0,'id_var':id_var}
	return fit_info

def get_core(R):
    centerX=0
    centerY=0
    r = R * np.sqrt(random.randint(0,1000)/1000.)
    theta = random.randint(0,1000)/1000.*2*np.pi
    core_x = centerX + r * np.cos(theta)
    core_y = centerY + r * np.sin(theta)
    #core_x = 0
    #core_y = 0
    return (core_x,core_y)


	
        		


