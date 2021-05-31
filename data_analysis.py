import cProfile
import re

import pickle
import integration_ME as ME
import numpy as np

import cProfile,pstats

from scipy import sparse
import matplotlib.pyplot as plt
from random import choices


Nmax = 13 # Maximum number of stators
ME.Nmax = Nmax # overriding Nmax in mE module for consistenct along files
thin = 1000 # number of datapoints to ignore between measurements
timeunits = 1 # timestep between measurements in the data
FPS = 1000 # frames per second 
factor_seconds = thin/FPS # seconds between timepoints
threshold_aver_stall = int(475 * FPS / thin) # number of points to include in the average trajectories (for plotting)
threshold_aver_release = int(825 * FPS / thin) # number of points to include in the average trajectories (for plotting)
threshold_aver_res = int(660 * FPS / thin) # number of points to include in the average trajectories (for plotting)
threshold_aver_res = int(660 * FPS / thin) # number of points to include in the average trajectories (for plotting)

file_dict = {300: "data/NilsD300_V2.p", 500: "data/NilsD500_V2.p", 1300: "data/NilsD1300_V2.p"} # files with the numeric traces

def _nonzero_mean(trajs, threshold = 3, excludezeros = True):
# given a set of aligned trajectories (rows) return the average of strictly positive entries only if there are at least 
# a threshold numnber of positive entries
# if excludezeros is True then the zeros are taken into account to do the average

	if excludezeros:
		accepted_entries = trajs>0
		mean = np.nansum(trajs,axis = 0)
		countnans = np.isnan(trajs).sum(axis = 0)

		countnonzero = np.count_nonzero(trajs, axis = 0)
		countaccepted = countnonzero - countnans 
		mean = mean/countaccepted
	else:
		mean = np.nanmean(trajs,axis = 0)
		countaccepted = (~np.isnan(trajs)).sum(axis = 0)		

	mean[countaccepted<threshold] = np.nan

	return mean

##################################################################################################
## The data is processed to be stored in a lighter format into the data_light dictionary
## by thining the data and calculating statistics

data_light = {300: [],500: [],1300: []}
data_light_stats = {300: {},500: {},1300: {}} # summary of some stats of data_light
for gamma in file_dict:

	max_len_stall = 0
	max_len_res_nozeros = 0
	max_len_res_withzeros = 0
	max_len_release = 0
	init_stall = np.zeros([])

	# this cumulative containers will be used temporaly to perform average and std deviations of all trajectories in a condition
	cumulative_trajectories_stall = np.array([],dtype = float).reshape(0,threshold_aver_stall)
	cumulative_trajectories_release = np.array([],dtype = float).reshape(0,threshold_aver_release) 
	cumulative_trajectories_res_nozeros = np.array([],dtype = float).reshape(0,threshold_aver_res) 
	cumulative_trajectories_res_withzeros = np.array([],dtype = float).reshape(0,threshold_aver_res) 
    
	dataset = pickle.load(open(file_dict[gamma],'rb'))
	data_light[gamma] = []
	data_light_stats[gamma]['resurrection_dwell'] = []
	for exp in dataset:
		dict_temp = {}
		if 'statnum_before_stall' in dataset[exp]:
			init_stall = np.append(init_stall,dataset[exp]['statnum_before_stall'][0])
			dd = np.array(dataset[exp]['statnum_before_stall'][::thin],dtype = float)
			dd[dd>Nmax] = Nmax # correcting upper limit to Nmax 
			dd[dd<0] = 0 # correcting bottom limit to 0
			non_zero_idx = np.argmax(dd>0) # will return the index with the first non-zero stators after resurrection (argmax stops at the first True)
			dd = dd[non_zero_idx:] # removing the first part of the transient where the number of stators is zero to synchronize all the resurrections
			dict_temp['statnum_before_stall'] = dd
			npoints = len(dict_temp['statnum_before_stall'])
			max_len_stall = max([max_len_stall, npoints])
			# to find an average trajectory for the condition the array is padded with nans to make a nanaverage,nanvar at the end
			cumulative_trajectories_stall = np.vstack((cumulative_trajectories_stall, 
				np.pad(dict_temp['statnum_before_stall'][:threshold_aver_stall],(0,max(0,threshold_aver_stall-npoints)),'constant',constant_values = np.nan)))
		else:
			print("No data found for load {}, replicate {}, condition statnum_before_stall".format(gamma,exp))

		if 'statnum_after_release' in dataset[exp]:
			dd = dataset[exp]['statnum_after_release'][::thin]
			dd[dd>Nmax] = Nmax # correcting limit to Nmax
			dd[dd<0] = 0 # correcting bottom limit to 0
			non_zero_idx = np.argmax(dd>0) # will return the index with the first non-zero stators after resurrection (argmax stops at the first True)
			dd = dd[non_zero_idx:] # removing the first part of the transient where the number of stators is zero to synchronize all the resurrections
			dict_temp['statnum_after_release'] = dd
			npoints = len(dict_temp['statnum_after_release'])
			max_len_release = max([max_len_release, len(dict_temp['statnum_after_release'])])
			cumulative_trajectories_release = np.vstack((cumulative_trajectories_release, 
				np.pad(dict_temp['statnum_after_release'][:threshold_aver_release],(0,max(0,threshold_aver_release-npoints)),'constant',constant_values = np.nan)))
		else:
			print("No data found for load {}, replicate {},condition statnum_after_release".format(gamma,exp))

		if 'statnum_resurrection' in dataset[exp]:
			dd = dataset[exp]['statnum_resurrection'][::thin]
			dd[dd>Nmax] = Nmax # correcting limit to Nmax
			dd[dd<0] = 0 # correcting bottoms limit to 0
			non_zero_idx = np.argmax(dd>0) # will return the index with the first non-zero stators after resurrection (argmax stops at the first True)
			if dd[non_zero_idx]<6:
				data_light_stats[gamma]['resurrection_dwell'].append(non_zero_idx)
				dict_temp['statnum_resurrection_withzeros'] = np.copy(dd)
				dd = dd[non_zero_idx:] # removing the first part of the transient where the number of stators is zero to synchronize all the resurrections
				dict_temp['statnum_resurrection_nozeros'] = dd 
				npoints_nozeros = len(dict_temp['statnum_resurrection_nozeros'])
				npoints_withzeros = len(dict_temp['statnum_resurrection_withzeros'])
				max_len_res_nozeros = max([max_len_res_nozeros, npoints_nozeros])
				max_len_res_withzeros = max([max_len_res_withzeros, npoints_withzeros])

				cumulative_trajectories_res_nozeros = np.vstack((cumulative_trajectories_res_nozeros, 
					np.pad(dict_temp['statnum_resurrection_nozeros'][:threshold_aver_res],(0,max(0,threshold_aver_res-npoints_nozeros)),
						'constant',constant_values = np.nan)))
				max_len_res_withzeros = max([max_len_res_withzeros, npoints_withzeros])

				cumulative_trajectories_res_withzeros = np.vstack((cumulative_trajectories_res_withzeros, 
					np.pad(dict_temp['statnum_resurrection_withzeros'][:threshold_aver_res],(0,max(0,threshold_aver_res-npoints_withzeros)),
						'constant',constant_values = np.nan)))
				max_len_res_nozeros = max([max_len_res_nozeros, npoints_nozeros])

		else:
			print("No data found for load {}, replicate {},condition statnum_resurrection".format(gamma,exp))

		data_light[gamma].append(dict(dict_temp))

	print('mean stall gamma = {}, = {}'.format(gamma,np.mean(init_stall)))

	# Storing trajectory stats per condition (mostly for plotting)
	data_light_stats[gamma]['longest_time_vector_stall'] = np.arange(max_len_stall)*factor_seconds
	data_light_stats[gamma]['longest_time_vector_resurrection_nozeros'] = np.arange(max_len_res_nozeros)*factor_seconds 
	data_light_stats[gamma]['longest_time_vector_resurrection_withzeros'] = np.arange(max_len_res_withzeros)*factor_seconds 
	data_light_stats[gamma]['longest_time_vector_release'] = np.arange(max_len_release)*factor_seconds 

	data_light_stats[gamma]['common_time_vector_stall'] = np.arange(threshold_aver_stall)*factor_seconds
	data_light_stats[gamma]['common_time_vector_resurrection_nozeros'] = np.arange(threshold_aver_res)*factor_seconds 
	data_light_stats[gamma]['common_time_vector_resurrection_withzeros'] = np.arange(threshold_aver_res)*factor_seconds 
	data_light_stats[gamma]['common_time_vector_release'] = np.arange(threshold_aver_release)*factor_seconds 

	cumulative_trajectories_stall[cumulative_trajectories_stall == 0] = np.nan
	cumulative_trajectories_release[cumulative_trajectories_release == 0] = np.nan

	data_light_stats[gamma]['mean_stall'] = np.nanmean(cumulative_trajectories_stall,axis = 0)
	data_light_stats[gamma]['mean_resurrection_nozeros'] = _nonzero_mean(cumulative_trajectories_res_nozeros)
	#print('cumulative_trajectories_res_nozeros',cumulative_trajectories_res_nozeros)
	data_light_stats[gamma]['mean_resurrection_withzeros'] = _nonzero_mean(cumulative_trajectories_res_withzeros)
	data_light_stats[gamma]['mean_resurrection_withzeros'][0] = 0 # the first zero is the only believable one
	print('mean_nozeros', gamma,data_light_stats[gamma]['mean_resurrection_nozeros'])
	print('mean_withzeros', gamma,data_light_stats[gamma]['mean_resurrection_withzeros'])


	data_light_stats[gamma]['mean_release'] = np.nanmean(cumulative_trajectories_release,axis = 0)

	data_light_stats[gamma]['median_stall'] = np.nanmedian(cumulative_trajectories_stall,axis = 0)
	data_light_stats[gamma]['median_resurrection_nozeros'] = np.nanmedian(cumulative_trajectories_res_nozeros,axis = 0)
	data_light_stats[gamma]['median_resurrection_withzeros'] = np.nanmedian(cumulative_trajectories_res_withzeros,axis = 0)
	data_light_stats[gamma]['median_release'] = np.nanmedian(cumulative_trajectories_release,axis = 0)

	data_light_stats[gamma]['std_stall'] = np.nanstd(cumulative_trajectories_stall,axis = 0)
	data_light_stats[gamma]['std_resurrection_nozeros'] = np.nanstd(cumulative_trajectories_res_nozeros,axis = 0)
	data_light_stats[gamma]['std_resurrection_withzeros'] = np.nanstd(cumulative_trajectories_res_withzeros,axis = 0)
	data_light_stats[gamma]['std_release'] = np.nanstd(cumulative_trajectories_release,axis = 0)

	data_light_stats[gamma]['mean_stall_distance'] = np.sum(data_light_stats[gamma]['std_stall'])
	data_light_stats[gamma]['mean_resurrection_nozeros_distance'] = np.sum(data_light_stats[gamma]['std_resurrection_nozeros'])
	data_light_stats[gamma]['mean_resurrection_withzeros_distance'] = np.sum(data_light_stats[gamma]['std_resurrection_withzeros'])
	data_light_stats[gamma]['mean_release_distance'] = np.sum(data_light_stats[gamma]['std_release'])


	print('mean distances gamma:',data_light_stats[gamma]['mean_stall_distance'],data_light_stats[gamma]['mean_release_distance'],data_light_stats[gamma]['mean_resurrection_nozeros_distance'])

#dataset = None	# empty this monster if there are memory issues
cumulative_trajectories_release = None
cumulative_trajectories_stall = None
cumulative_trajectories_res_nozeros = None
cumulative_trajectories_res_withzeros = None

def DistanceLikelihood(gamma = 300, model = 'WeakStrong', params = [], return_trajs = False, sto_trajs = False, resurrectionzeros = 'nozeros',
					   likelihoodcondition = 'all'):
# for each model and parameters, create a compatible set of average trajectories and compare them to 
# the average trajectories measured
# if sto_trajs is True the likelihood corresponds to one trajectory, otherwise the mean is used
# if the mean is used then WeakStrong is integrated using 2-D the analytical solution rather than solving the N^2 ME
# resurrection can be 'nozeros' or 'withzeros' to consider that alignment of the resurrections.
# 'withzeros': the resurrections starts at t=0
# 'nonzeros' : the starting time and occupancy are the first stator recruitment in the resurrection trace
# likelihoodconditions can be 'all', or a list with the selected conditions 'stall', 'release', 'resurrection'
# models can be 'Berg' for the speed-rate model proposed in Wadhwa et al 2019 or 'WeakStrong' for the two-state model 

	if resurrectionzeros == 'nozeros':
		str_res = 'statnum_resurrection_nozeros'
		str_common_vector_res = 'common_time_vector_resurrection_nozeros'
		str_mean_res = 'mean_resurrection_nozeros'
	elif resurrectionzeros == 'withzeros' :
		str_res = 'statnum_resurrection_withzeros'
		str_common_vector_res = 'common_time_vector_resurrection_withzeros'
		str_mean_res = 'mean_resurrection_withzeros'

	if likelihoodcondition == 'all':
		likelihoodcondition = ['stall','release','resurrection']

	if model == 'Berg':
		A = ME.Berg_Matrix(*params, N=Nmax)	
		Aext = ME.GetEigenElements(A)
		P_eq = ME.Normalize(ME.Equilibrium(Aext,eigen_given = True))
	elif model == 'WeakStrong':
		if sto_trajs:
			# full ME is only required when comparing stochastic trajectories
			A = ME.WeakStrong_Matrix(*params, N=Nmax)
			Aext = ME.GetEigenElements(A)
			P_eq = ME.Normalize(ME.Equilibrium(Aext,eigen_given = True))
		params_stall = params[:]
		params_stall[1] = 0 # forbidden dettachment 
		A_stall = ME.WeakStrong_Matrix(*params_stall, N=Nmax)
		Aext_stall = ME.GetEigenElements(A_stall)
		P_eq_stall = ME.Normalize(ME.Equilibrium(Aext_stall,eigen_given = True))

	if model == 'Berg':
		cumulative_trajectories_stall = np.zeros_like(data_light_stats[gamma]['common_time_vector_stall'])
		cumulative_trajectories_release = np.zeros_like(data_light_stats[gamma]['common_time_vector_release'])
		cumulative_trajectories_res = np.zeros_like(data_light_stats[gamma][str_common_vector_res])


	elif model == 'WeakStrong':
		cumulative_trajectories_stall = np.zeros((len(data_light_stats[gamma]['common_time_vector_stall']),2))
		cumulative_trajectories_release = np.zeros((len(data_light_stats[gamma]['common_time_vector_release']),2))
		cumulative_trajectories_res = np.zeros((len(data_light_stats[gamma][str_common_vector_res]),2))

	N_stall = 0 # number of trajectories acceptee
	N_release = 0
	N_res = 0

	for exp in data_light[gamma]: # for each experimental trajectory, the initial condition is used to integrate the equations
	# afterwards the summary statistics are calculated with the ensemble of trajectories and compared to the experiments

		#### BEFORE STALL (STEADY STATE)

		if 'stall' in likelihoodcondition:
			N0_stall = exp['statnum_before_stall'][0].astype(int) #
			if model == 'Berg':
				if sto_trajs is True:
					cumulative_trajectories_stall += ME.Get_Berg_Trajectory(N0_stall, *params,
						data_light_stats[gamma]['common_time_vector_stall'])
				else:
					P0_before_stall = np.zeros_like(P_eq)
					P0_before_stall[N0_stall] = 1 # initial condition
					Pt_before_stall = ME.Integrate(Aext, P0_before_stall, data_light_stats[gamma]['common_time_vector_stall'], eigen_given= True) 
					mean_stall, std_stall = ME.stats(Pt_before_stall,model,Nmax)
					cumulative_trajectories_stall += mean_stall

			elif model == 'WeakStrong':
				if sto_trajs is True:
					W0_stall,S0_stall = ME.WeakStrong_indextovalues(choices(range(len(P_eq)),P_eq)[0])
					cumulative_trajectories_stall += ME.Get_WeakStrong_Trajectory(W0_stall, S0_stall,
						*params, data_light_stats[gamma]['common_time_vector_stall'])
				else:
					# These lines solve the ME for initialm condition. Not required at the moment, just using the mean number
					#P0_before_stall = np.copy(P_eq)
					#ME.ConditionPwstoN(P0_before_stall, N0_stall)
					#Pt_before_stall = ME.Integrate(Aext, P0_before_stall, data_light_stats[gamma]['common_time_vector_stall'], eigen_given= True) 
					#mean_stall, std_stall, w_stall, s_stall = ME.stats(Pt_before_stall,model,Nmax, weakstrong_output = True)

					w_stall, s_stall = ME.MeanSteadyStateWeakStrong(*params)
					cumulative_trajectories_stall[:,0] += w_stall
					cumulative_trajectories_stall[:,1] += s_stall
				
			N_stall += 1

		####  RELEASE
		# The release initial points are taken from each individual trajectory
		if 'release' in likelihoodcondition:
			N0_release = exp['statnum_after_release'][0].astype(int)
			if model == 'Berg':
				if sto_trajs is True:
					cumulative_trajectories_release += ME.Get_Berg_Trajectory(N0_release, *params,
						data_light_stats[gamma]['common_time_vector_release'])
				else:
					P0_after_release = np.zeros_like(P_eq)
					P0_after_release[N0_release] = 1
					Pt_after_release = ME.Integrate(Aext, P0_after_release, data_light_stats[gamma]['common_time_vector_release'], eigen_given= True) 
					mean_rel, std_rel = ME.stats(Pt_after_release,model,Nmax)
					cumulative_trajectories_release += mean_rel

			elif model == 'WeakStrong':
			# In the weakstorng model the intial weak strong stators are taken randomly 
			# from the stall steady state distribution (kwu=0) and the condition (W+S =N0)
				if sto_trajs is True:				
					P_rel = np.copy(P_eq_stall)
					ME.ConditionPwstoN(P_rel,N0_release)
					W0_release,S0_release = ME.WeakStrong_indextovalues(choices(range(len(P_rel)),P_rel)[0])
					cumulative_trajectories_release += ME.Get_WeakStrong_Trajectory(W0_release, S0_release,
					 	*params, data_light_stats[gamma]['common_time_vector_release'])
				else:
					P0_after_release = np.copy(P_eq_stall)
					ME.ConditionPwstoN(P0_after_release, N0_release)

					### These two lines would integrate the equation numerically (keeping for future comparisons with std)
					# Pt_after_release = ME.Integrate(Aext, P0_after_release, data_light_stats[gamma]['common_time_vector_release'], eigen_given= True) 
					# mean_rel_old, std_rel_old, mean_rel_weak_old, mean_rel_strong_old = ME.stats(Pt_after_release,model,Nmax, weakstrong_output = True)

					N0, std0, w0, s0 = ME.stats(P0_after_release,model,Nmax,weakstrong_output = True)
					mean_rel_weak, mean_rel_strong = ME.TrajMeanFieldWeakStrong(*params, w0, s0, data_light_stats[gamma]['common_time_vector_release'],
						                             Nmax=Nmax, method = 'analytical')
					mean_rel = mean_rel_weak + mean_rel_strong
					cumulative_trajectories_release[:,0] += mean_rel_weak
					cumulative_trajectories_release[:,1] += mean_rel_strong

			N_release += 1	

		if 'resurrection' in likelihoodcondition:
			if str_res in exp: ## Not all the datasets have resurrection traces

				N0_resurrection = exp[str_res][0].astype(int)

				if model == 'Berg':
					if sto_trajs is True:
						cumulative_trajectories_res += ME.Get_Berg_Trajectory(N0_resurrection, *params,
							data_light_stats[gamma][str_common_vector_res])
					else:
						P0_resurrection = np.zeros_like(P_eq)
						P0_resurrection[N0_resurrection] = 1
						Pt_resurrection = ME.Integrate(Aext, P0_resurrection, data_light_stats[gamma][str_common_vector_res], eigen_given= True) 
						mean_res, std_res = ME.stats(Pt_resurrection,model,Nmax)
						cumulative_trajectories_res += mean_res

				elif model == 'WeakStrong':
				# In the weakstorng model the intial stators are assumed to be bound weakly
					if sto_trajs is True:
						cumulative_trajectories_res += ME.Get_WeakStrong_Trajectory(N0_resurrection, 0, *params,
							data_light_stats[gamma][str_common_vector_res])
					else:
						P0_resurrection = np.zeros_like(P_eq_stall)
						P0_resurrection[ME.WeakStrong_valuestoindex(N0_resurrection,0)] = 1

						### These two lines would integrate the equation numerically (keeping for future comparisons with std)
						# Pt_resurrection = ME.Integrate(Aext, P0_resurrection, data_light_stats[gamma][str_common_vector_res], eigen_given= True) 
						# mean_res, std_res, mean_res_weak, mean_res_strong = ME.stats(Pt_resurrection,model,Nmax, weakstrong_output = True)

						mean_res_weak, mean_res_strong = ME.TrajMeanFieldWeakStrong(*params, 0, 0, data_light_stats[gamma][str_common_vector_res],
						                             Nmax=Nmax, method = 'analytical')
						cumulative_trajectories_res[:,0] += mean_res_weak
						cumulative_trajectories_res[:,1] += mean_res_strong

				N_res += 1	

	if model == 'WeakStrong':
		# mean number of stators in the two-state model is the sum of weak and strongly bound stators
		cumulative_trajectories_stall = cumulative_trajectories_stall[:,0]+cumulative_trajectories_stall[:,1]
		cumulative_trajectories_release = cumulative_trajectories_release[:,0]+cumulative_trajectories_release[:,1]
		cumulative_trajectories_res = cumulative_trajectories_res[:,0]+cumulative_trajectories_res[:,1]

	distance = 0 
	if 'stall' in likelihoodcondition:
		cumulative_trajectories_stall/= N_stall
		distance_stall = cumulative_trajectories_stall - data_light_stats[gamma]['mean_stall']
		distance_stall = np.sum(distance_stall*distance_stall)	
		distance += distance_stall
	if 'resurrection' in likelihoodcondition:
		cumulative_trajectories_res/= N_res
		distance_res = cumulative_trajectories_res - data_light_stats[gamma][str_mean_res]
		distance_res = np.nansum(distance_res*distance_res)	
		distance += distance_res
	if 'release' in likelihoodcondition:
		cumulative_trajectories_release/= N_release
		distance_release = cumulative_trajectories_release - data_light_stats[gamma]['mean_release']
		distance_release = np.sum(distance_release*distance_release)
		distance += distance_release

	if return_trajs:
		return(cumulative_trajectories_stall,cumulative_trajectories_release,cumulative_trajectories_res,distance)
	else:

		return(distance)
	

	
