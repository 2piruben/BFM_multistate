import data_analysis as ed
import integration_ME as ME
import matplotlib.pyplot as plt
from matplotlib import gridspec, cm
import numpy as np
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
import models as md

Nmax = ed.Nmax

sbcolorcycle=sns.color_palette("Set1",n_colors=9,desat=0.8) # Setting Colour palette
sns.set_palette("Set1",n_colors=9,desat=0.8)
sbcolorcyclebright=sns.color_palette("bright")
sbcolorcycledark=sns.color_palette("dark")
sbcolorcyclemuted=sns.color_palette("muted")
sns.set_style("ticks") # Set plot properties
sns.despine()

colorcycle=["#e41a1c", #RED
  "#377eb8","#4daf4a", #BLUE, GREEN
  "#ff7f00","#984ea3", #ORANGE, PURPLE
  "#999999","#f781bf","#a65628","#ffff33"] #GREY, PINK, BROWN

def plottrajectory(model = 'BergU', bayes = 'SMC', gamma = 300, source = 'file',
                  condition = 'final', bestpar_idx = 'MAP', resurrection = 'withzeros', subfolder = 'smc/'):
# plots the experimental trajectories, their statistics and the prediction of the model for a given set of parameters
# if 'source'== pars then the parameters are the array pars, if 'file' exract from fitting 
# if 'bestpar_idx' == 'MAP' then the kde is usedd to calculate the MAP and plot that trajectory, otherwise use int to select a dataset
# model : what model from models.py to use
# bayes : currently only using 'SMC' - sequential monte carlo
# gamma : value of the load
# condition: what iteration of the SMC to choose from. 'final' returns the last one
# resurrection: choose alignment of resurrectiont races ca ben 'withzeros' or 'nozeros'

    stride_plots = 1 # stride even more to have less crammed correlation plots
    condition = str(condition)
    fontsize = 15

    if resurrection == 'nozeros':
        str_res = 'statnum_resurrection_nozeros'
        str_common_vector_res = 'common_time_vector_resurrection_nozeros'
        str_longest_vector_res = 'longest_time_vector_resurrection_nozeros'
        str_mean_res = 'mean_resurrection_nozeros'

    elif resurrection == 'withzeros' :
        str_res = 'statnum_resurrection_withzeros'
        str_common_vector_res = 'common_time_vector_resurrection_withzeros'
        str_longest_vector_res = 'longest_time_vector_resurrection_withzeros'
        str_mean_res = 'mean_resurrection_withzeros'



    if md.model_properties[model]['integrator_name']=='Berg':
        N_dimension = Nmax+1
            
    elif md.model_properties[model]['integrator_name']=='WeakStrong':
        parnames = []
        N_dimension = int((Nmax+1)*(Nmax+2)/2)


    data1 = None
    lklh_list = np.array([])

    if source == 'file':
        if bayes == 'SMC': # only option in the last implementation
            # gamma_file = gamma
            # if model in ['BergU','BergO','BergOO','WeakStrongU_fixstrong','WeakStrongU','WeakStrongU_doublefix']:
            gamma_file = 300
            data1_sorted = np.loadtxt('{}pars_{}_MF_{}_{}.out'.format(subfolder,model,gamma_file,condition))

            if bestpar_idx == 'MAP': # calculate MAP
                kde = gaussian_kde(data1_sorted.T, bw_method = 1.0)
                localdensity = kde.evaluate(data1_sorted.T)
                sortedidxs = localdensity.argsort()
                data1_sorted = data1_sorted[sortedidxs]
                localdensity_sorted = localdensity[sortedidxs]
                bestpars = data1_sorted[-1]
                worstpars = data1_sorted[0]
                print("Best pars for the stochastic model are: {}".format(bestpars))
                print("Worst pars for the stochastic model are: {}".format(worstpars))
                distance = 0
                gammas = [300,500,1300]
                integrator_name = md.model_properties[model]['integrator_name']
                sto_trajs = False
                resurrectionzeros = md.model_properties[model]['resurrectionzeros']
                for xgamma in gammas:
                    integration_pars = md.prior_to_integrator(bestpars,model,xgamma)
                    distance += ed.DistanceLikelihood(xgamma, integrator_name, integration_pars,
                        return_trajs = False, sto_trajs = False,resurrectionzeros = resurrectionzeros)
                bestpars = md.prior_to_integrator(bestpars,model,gamma)    
                print("Distance at MAP is: {}".format(distance))

            elif bestpar_idx == 'score': # calculate the score of all the points and select the best one (this might take time)

                gammas = [300,500,1300]
                distances = []
                integrator_name = md.model_properties[model]['integrator_name']
                resurrectionzeros = md.model_properties[model]['resurrectionzeros']
                distances = np.loadtxt('{}distances_{}_MF_{}_{}.out'.format(subfolder,model,gamma_file,condition))

                # for par in tqdm(data1_sorted):
                #     distance = 0
                #     for xgamma in gammas:
                #         integration_pars = md.prior_to_integrator(par,model,xgamma)
                #         distance += ed.DistanceLikelihood(xgamma, integrator_name, integration_pars,
                #             return_trajs = False, sto_trajs = False,resurrectionzeros = resurrectionzeros)
                #     distances.append(distance)

                bestidx = np.argmin(distances)
                worstidx = np.argmax(distances)
                bestpars = md.prior_to_integrator(data1_sorted[bestidx],model,gamma) 
                worstpars = md.prior_to_integrator(data1_sorted[worstidx],model,gamma) 
                print("Best sample distance is: {}".format(distances[bestidx]))
                print("For parameter set:", data1_sorted[bestidx])
                print("Worst sample distance is: {}".format(distances[worstidx]))

            else:
                bestpars = data1_sorted[bestpar_idx]
                bestpars = md.prior_to_integrator(bestpars,model,gamma)
                print("Particular point chosen : {}".format(bestpars))
    else:
        print('source', source)
        bestpars = md.prior_to_integrator(source,model,gamma)
        print("Using custom parameter set (in sampling space):", source)
        distance = 0 
        gammas = [300,500,1300]
        integrator_name = md.model_properties[model]['integrator_name']
        resurrectionzeros = md.model_properties[model]['resurrectionzeros']
        for xgamma in gammas:
            integration_pars = md.prior_to_integrator(source,model,xgamma)
            distance += ed.DistanceLikelihood(xgamma, integrator_name, integration_pars,
                return_trajs = False, sto_trajs = False,resurrectionzeros = resurrectionzeros)
        print("With distance: ", distance)

    ## preparing figures and axes
    fig = plt.figure(figsize = [15,5])
    spec = gridspec.GridSpec(ncols=3, nrows=1, figure=fig, wspace = 0.01)
    axes = []
    for expcondition in range(3):
        axes.append(plt.subplot(spec[expcondition]))
        if expcondition > 0 :
            axes[expcondition].yaxis.set_ticklabels([])

    ################################ Plotting trajectories
    # Getting the stochastic trajectories

    if md.model_properties[model]['integrator_name'] == 'Berg':
            A = ME.Berg_Matrix(*bestpars, N=Nmax)
            Aext = ME.GetEigenElements(A) # to avoid inverting the matrix constantly
            P0_before_stall = ME.Equilibrium(Aext, eigen_given = True)
            P0_resurrection = np.zeros(Nmax+1)
            P0_resurrection[0] = 1

    elif md.model_properties[model]['integrator_name'] == 'WeakStrong':
            A = ME.WeakStrong_Matrix(*bestpars, N=Nmax)
            Aext = ME.GetEigenElements(A) # to avoid inverting the matrix constantly
            P0_before_stall = ME.Equilibrium(Aext, eigen_given = True)

            params_stall = bestpars[:]
            params_stall[1] = 0 # forbidden dettachment 
            A_stall = ME.WeakStrong_Matrix(*params_stall, N=Nmax)
            Aext_stall = ME.GetEigenElements(A_stall)
            P_eq_stall = ME.Normalize(ME.Equilibrium(Aext_stall,eigen_given = True))

            M = len(P0_before_stall) # number of possible states
            P0_resurrection = np.zeros(M)
            P0_resurrection[0] = 1 # first state is (empty state) is the only populated one

        ## Due to the fact that the initial condition for stalls is not fixed, the statistics are computed using all the trajectories

    Pt_release_cumulative = np.zeros((len(ed.data_light_stats[gamma]['longest_time_vector_release']),N_dimension))
    N_release = 0
    Pt_resurrection_cumulative = np.zeros((len(ed.data_light_stats[gamma][str_longest_vector_res]),N_dimension))

    # each one will contain the mean and var of the different release traces (due to different initial conditions)
    mean_release = np.array([]).reshape(0,len(ed.data_light_stats[gamma]['longest_time_vector_release'])) 
    mean_release_weak = np.array([]).reshape(0,len(ed.data_light_stats[gamma]['longest_time_vector_release'])) 
    mean_release_strong = np.array([]).reshape(0,len(ed.data_light_stats[gamma]['longest_time_vector_release'])) 
    var_release = np.array([]).reshape(0,len(ed.data_light_stats[gamma]['longest_time_vector_release']))
    mean_resurrection = np.array([]).reshape(0,len(ed.data_light_stats[gamma][str_longest_vector_res])) 
    mean_resurrection_weak = np.array([]).reshape(0,len(ed.data_light_stats[gamma][str_longest_vector_res])) 
    mean_resurrection_strong= np.array([]).reshape(0,len(ed.data_light_stats[gamma][str_longest_vector_res])) 
    var_resurrection = np.array([]).reshape(0,len(ed.data_light_stats[gamma][str_longest_vector_res]))

    for exp in ed.data_light[gamma]:
        axes[0].plot(np.arange(len(exp['statnum_before_stall']))*ed.factor_seconds,exp['statnum_before_stall'],'k',alpha = 0.2, zorder = -1)
        axes[1].plot(np.arange(len(exp['statnum_after_release']))*ed.factor_seconds,exp['statnum_after_release'],'k',alpha = 0.2, zorder = -1)

        ### Calculating release trajectories
        if md.model_properties[model]['integrator_name'] == 'Berg':
            P0_after_release = np.zeros(Nmax+1)
            N_empiric_stall = int(min(Nmax,exp['statnum_after_release'][0]))
            P0_after_release[N_empiric_stall] = 1
            P_rel = np.copy(P0_after_release)                              

        elif md.model_properties[model]['integrator_name'] == 'WeakStrong':
            P_rel = np.copy(P_eq_stall)                
            N0_release = int(min(Nmax,exp['statnum_after_release'][0]))
            ME.ConditionPwstoN(P_rel,N0_release)

        Pt = ME.Integrate(Aext,P_rel, ed.data_light_stats[gamma]['longest_time_vector_release'], eigen_given= True) 
        tolP = 1E-6
        Pt = np.abs(Pt+tolP)# adding a background to discard equally improbable events, abs discards negative small probability values
        Pt = Pt/np.sum(Pt,axis = 1)[:,np.newaxis]# renormalizing
        Pt_release_cumulative += Pt
        N_release += 1

        if md.model_properties[model]['integrator_name'] == 'Berg':
                mean_rel, std_rel = ME.stats(Pt,md.model_properties[model]['integrator_name'],Nmax)
        elif md.model_properties[model]['integrator_name'] == 'WeakStrong':
                mean_rel, std_rel, mean_rel_weak, mean_rel_strong = ME.stats(Pt,md.model_properties[model]['integrator_name'],Nmax,weakstrong_output = True)
                mean_release_weak = np.vstack((mean_release_weak,mean_rel_weak))
                mean_release_strong = np.vstack((mean_release_strong,mean_rel_strong))    
        mean_release = np.vstack((mean_release,mean_rel))
        var_release = np.vstack((var_release,std_rel*std_rel))

        ### Calculating resurrection trajectories
        if str_res in exp:

            axes[2].plot(np.arange(len(exp[str_res]))*ed.factor_seconds,exp[str_res],'k',alpha = 0.2, zorder = -1)

            if md.model_properties[model]['integrator_name'] == 'Berg':
                P0_resurrection = np.zeros(Nmax+1)
                N_empiric_res = int(min(Nmax,np.abs(exp[str_res][0])))
                print('Initial coordinate for resurrection',N_empiric_res)
                P0_resurrection[N_empiric_res] = 1
                
            if md.model_properties[model]['integrator_name'] == 'WeakStrong':
                P0_resurrection = np.zeros(M)
                N_empiric_stall = int(min(Nmax,exp[str_res][0]))
                initial_state_idx = ME.WeakStrong_valuestoindex(N_empiric_stall,0,N = Nmax) # assuming all the stators are weakly bound
                P0_resurrection[initial_state_idx] = 1

            Pt = ME.Integrate(Aext, P0_resurrection, ed.data_light_stats[gamma][str_longest_vector_res], eigen_given= True) 

            if md.model_properties[model]['integrator_name'] == 'Berg':
                mean_res, std_res = ME.stats(Pt,'Berg',Nmax)
            if md.model_properties[model]['integrator_name'] == 'WeakStrong':
                mean_res, std_res, mean_res_weak, mean_res_strong = ME.stats(Pt,'WeakStrong',Nmax,weakstrong_output = True)
                mean_resurrection_weak = np.vstack((mean_resurrection_weak,mean_res_weak))
                mean_resurrection_strong = np.vstack((mean_resurrection_strong,mean_res_strong))
    
            mean_resurrection = np.vstack((mean_resurrection,mean_res))
            var_resurrection = np.vstack((var_resurrection,std_res*std_res))


    # the mean for release traces is the mean of the means
    # print(mean_release)
    total_mean_release = np.mean(mean_release,axis = 0)
    total_mean_release_weak = np.mean(mean_release_weak,axis = 0)
    total_mean_release_strong = np.mean(mean_release_strong,axis = 0)
    # for the variance we need to use the total law of variance
    #total_var_release = np.var(mean_release,axis=0) + np.mean(var_release,axis=0)
    #total_std_release = np.sqrt(total_var_release)

    total_mean_resurrection = np.mean(mean_resurrection,axis = 0)
    total_mean_resurrection_weak = np.mean(mean_resurrection_weak,axis = 0)
    total_mean_resurrection_strong = np.mean(mean_resurrection_strong,axis = 0)

    # for the variance we need to use the total law of variance
    total_var_resurrection = np.var(mean_resurrection,axis=0) + np.mean(var_resurrection,axis=0)
    total_std_resurrection = np.sqrt(total_var_resurrection)

    axes[1].plot(ed.data_light_stats[gamma]['longest_time_vector_release'], total_mean_release,color = 'b', zorder = 100)
    axes[1].plot(ed.data_light_stats[gamma]['longest_time_vector_release'], total_mean_release_weak,color = 'c', zorder = 100)
    axes[1].plot(ed.data_light_stats[gamma]['longest_time_vector_release'], total_mean_release_strong,color = 'm', zorder = 100)
    print("Initial condition release ", total_mean_release_weak[0],total_mean_release_strong[0])

    # axes[1].fill_between(ed.data_light_stats[gamma]['longest_time_vector_release'],
    #     total_mean_release-total_std_release, total_mean_release+total_std_release,color = 'b', alpha = 0.5)
    axes[1].set_ylim([0,13])

    ## Plotting resurrection traces
    # Pt_resurrection = ME.Integrate(Aext, P0_resurrection, ed.data_light_stats[gamma][str_longest_vector_res], eigen_given= True) 
    # mean,std = ME.stats(Pt_resurrection,model,Nmax)
    axes[2].plot(ed.data_light_stats[gamma][str_longest_vector_res], total_mean_resurrection,color = 'b', zorder = 100)
    axes[2].plot(ed.data_light_stats[gamma][str_longest_vector_res], total_mean_resurrection_weak,color = 'c', zorder = 100)
    axes[2].plot(ed.data_light_stats[gamma][str_longest_vector_res], total_mean_resurrection_strong,color = 'm', zorder = 100)
    # axes[2].fill_between(ed.data_light_stats[gamma][str_longest_vector_res],
    #     total_mean_resurrection-total_std_resurrection, total_mean_resurrection+total_std_resurrection,color = 'b', alpha = 0.5)
    axes[2].set_ylim([0,13])
    axes[2].set_xlim([0,660])

    ## Plotting steady-state traces
    Pt_before_stall = ME.Integrate(Aext, P0_before_stall, ed.data_light_stats[gamma]['longest_time_vector_stall'], eigen_given= True) 
    if md.model_properties[model]['integrator_name'] == 'Berg':
        mean_stall, std_stall = ME.stats(Pt_before_stall,'Berg',Nmax)
        axes[0].plot(ed.data_light_stats[gamma]['longest_time_vector_stall'], mean_stall,color = 'b', label = '$\\gamma = {}$'.format(gamma))

    else:
        mean_stall, std_stall, mean_stall_weak, mean_stall_strong = ME.stats(Pt_before_stall, 'WeakStrong', Nmax, weakstrong_output=True)
        axes[0].plot(ed.data_light_stats[gamma]['longest_time_vector_stall'], mean_stall,color = 'b', label = ('$\\langle N \\rangle_\\mathrm{theo}\\qquad\\gamma' + ' = {}$'.format(gamma)))
        axes[0].plot(ed.data_light_stats[gamma]['longest_time_vector_stall'], mean_stall_weak,color = 'c', label = '$\\langle w \\rangle$' )
        axes[0].plot(ed.data_light_stats[gamma]['longest_time_vector_stall'], mean_stall_strong,color = 'm', label = '$\\langle s \\rangle$' )
        print("Initial condition steady state ", mean_stall_weak[0],mean_stall_strong[0])

    # axes[0].fill_between(ed.data_light_stats[gamma]['longest_time_vector_stall'], mean-std, mean+std,color = 'b', alpha = 0.5)
    axes[0].set_ylim([0,13])

    # plotting average experimental trajectories
    axes[0].plot(ed.data_light_stats[gamma]['common_time_vector_stall'],ed.data_light_stats[gamma]['mean_stall'],'k-',lw = 1.5,label= '$\\langle N \\rangle_\\mathrm{exp}$', zorder = 200)
    axes[1].plot(ed.data_light_stats[gamma]['common_time_vector_release'],ed.data_light_stats[gamma]['mean_release'],'k-',lw = 1.5, zorder = 200)
    axes[2].plot(ed.data_light_stats[gamma][str_common_vector_res],ed.data_light_stats[gamma][str_mean_res],'k-',lw = 1.5, zorder = 200)

    axes[0].legend(bbox_to_anchor=(0., 1.02, 3., .102), loc=3, ncol=4, mode="expand", borderaxespad=0., fontsize = fontsize)
    axes[0].set_xlabel('time')
    axes[1].set_xlabel('time')
    axes[2].set_xlabel('time')
    axes[0].set_ylabel('occupancy')
    plt.savefig('results_trajs_{}_{}.png'.format(gamma,model), bbox_inches = 'tight')
    plt.show()



def plotpargrid(model = 'BergU', bayes = 'SMC', method = 'ME',gamma = 300, chain = 'all',
    ordermethod = 'logps', condition = 'final', figsize = 10, parset = 'all', color = 'KDE', subfolder = 'smc/'):

    '''
    Plot a grid with the results of the MCMC or SMC

    bayes --- MCMC or SMC
    chain --- which chains to use. 'all' uses all the chains
    ordermethod --- There are two different ways of ordering the goodness of the results. 
    One of them is compute one by one the likelihoods for each parset
    This methods can be quite slow sometimes. The second option is to load the logps from the output of pyDream
    condition --- Which output file to read (condition refers to number of samples from the MCMC)
    figsize --- size of figure in inches
    plotaffinities --- just plot the marginal distributions of the affinities
    color = 
    '''

    ntrim = 1000 # trim the transient of the MCMC
    stride = 5 # less points to avoid correlation between subsequent points
    stride_plots = 1 # stride even more to have less crammed correlation plots
    condition = str(condition)
    chain = chain
    scattersize = 5
    fontsize = 14
    #lklh_ci = 0.9 ## confidence interval threshold to plot 

    parnames = md.model_properties[model]['parnames']

    if 'gamma_file' in md.model_properties[model]:
        gamma_file = md.model_properties[model]['gamma_file']
    else:
        gamma_file = gamma
    lower_limits = md.model_properties[model]['lower_limits']
    scale_limits = md.model_properties[model]['scale_limits']
    axislim = np.array([lower_limits,lower_limits+scale_limits]).T
    axis_ticks = []
    tick_labels = []
    int_to_str = lambda x: str(int(x)) 
    for par in axislim:
        ticks = np.arange(par[0],par[1])
        axis_ticks.append(ticks)
        tick_labels.append(list(map(int_to_str,ticks)))


    data1 = np.array([]).reshape(0,len(parnames))
    lklh_list = np.array([])

    if bayes == 'MCMC':

        if chain == 'all':
          chainlist = [0,1,2]
        else:
          chainlist = [chain]      
        for ch in chainlist:
          data1_temp = np.load('dream/{}_{}_{}_fit_sampled_params_chain{}_{}.npy'.format(model,method,gamma,ch,condition))
          print('loading file: ',data1_temp)
          logpsdata1_temp = np.load('dream/{}_{}_{}_fit_logps_chain{}_{}.npy'.format(model,method,gamma,ch,condition))
          lklh_list_temp = np.array([l[0] for l in logpsdata1_temp])
          data1_temp = data1_temp[ntrim::stride]
          lklh_list_temp = lklh_list_temp[ntrim::stride]
          data1 = np.vstack((data1,data1_temp))
          lklh_list = np.concatenate((lklh_list,lklh_list_temp))


        if ordermethod == 'logps':

            sorted_idxs = lklh_list.argsort()
            data1_sorted = data1[sorted_idxs]
            lklh_list_sorted = lklh_list[sorted_idxs]
            bestlklh = lklh_list[-1]
            worstlklh = lklh_list[0]
            bestpars = data1_sorted[-1]
            ibestlklh = -1


    elif bayes == 'SMC':
        gamma_file = gamma
        #if model in ['BergU','BergO','WeakStrongU','WeakStrongU_fixstrong','WeakStrongU_doublefix']:
        gamma = 300
        data1 =  np.loadtxt('{}pars_{}_MF_{}_{}.out'.format(subfolder,model,gamma,condition))
        distances = np.loadtxt('{}distances_{}_MF_{}_{}.out'.format(subfolder,model,gamma,condition))
        print('Distance range:', min(distances), max(distances))
        kde = gaussian_kde(data1.T, bw_method = 2.0)
        localdensity = kde.evaluate(data1.T)
        sortedidxs = localdensity.argsort()
        data1_sorted = data1[sortedidxs]
        localdensity_sorted = localdensity[sortedidxs]

        print('######  SUMMARY HIGH DENSITY INTERVALS')
        print('MAP: {}'.format(data1_sorted[-1]))
        print('With score: {}'.format(distances[sortedidxs][-1]))
        idx_minscore = np.argmin(distances)
        print('Optimum par:{}'.format(data1[idx_minscore]))
        print('With score: {}'.format(distances[idx_minscore]))
        cutoff = len(data1_sorted)//10
        print('cutoff',cutoff)
        for ipar,par in enumerate(parnames):
            print(par)
            print('MAP:{}   (10**):{}'.format(data1_sorted[-1,ipar],10**data1_sorted[-1,ipar]))
            print('bestpar{}   (10**):{}:'.format(data1[idx_minscore,ipar],10**data1[idx_minscore,ipar]))
            print('min/max: [{},{}]    (10**):[{},{}]:'.format(min(data1_sorted[:,ipar]),max(data1_sorted[:,ipar]),10**min(data1_sorted[:,ipar]),10**max(data1_sorted[:,ipar]) ) )
            print('HDI(90%): [{},{}]    (10**):[{},{}]:'.format(min(data1_sorted[cutoff:,ipar]),max(data1_sorted[cutoff:,ipar]),10**min(data1_sorted[cutoff:,ipar]),10**max(data1_sorted[cutoff:,ipar]) ) )

        # Info of ratios of parameters
        # newpar_min =min(10**data1_sorted[cutoff:,4]/10**data1_sorted[cutoff:,5])
        # newpar_max =max(10**data1_sorted[cutoff:,4]/10**data1_sorted[cutoff:,5])
        # print('HDI ratio pars 4,5 is:',newpar_min,newpar_max)
        # print('MAP ratio pars 4,5 is:',10**data1_sorted[-1,4]/10**data1_sorted[-1,5])


    data_df1 = pd.DataFrame(data = data1_sorted, columns = parnames)
    if bayes == 'MCMC':
        data_df1['color'] = lklh_list_sorted
        vmin = min(data_df1['color'][::stride_plots])
        vmin = max(data_df1['color'][::stride_plots])
    elif bayes == 'SMC': 
        if color == 'KDE':
            data_df1['color'] = localdensity_sorted
            vmin = min(localdensity_sorted)
            vmax = max(localdensity_sorted)
        else:
            data_df1['color'] = 1.0
            vmin = 0
            vmax = 1

    #cmap = cm.get_cmap('twilight')
    #cmap = cmap_redblue
    cmap = cm.get_cmap('viridis')

    fig = plt.figure(figsize = (figsize,figsize))
    gs = gridspec.GridSpec(len(parnames), len(parnames))
    gaxes = []


    # Plot each subplot in coordinates irow,icol in a grid of plots
    for irow,row in enumerate(parnames):
        grow = []
        for icol,col in enumerate(parnames):
            if icol<=irow:
                grow.append(plt.subplot(gs[irow,icol]))
        gaxes.append(grow)  

    # Plot of offdiagonal scatter plots

    print("Plotting results for model {} with parnames {}".format(model,parnames))
    for irow,rowname in tqdm(enumerate(parnames)):
        for icol,colname in enumerate(parnames):
          # print(irow,icol)
          ## scatter plots
          if icol < irow:
            gaxes[irow][icol].set_facecolor(cmap(0))
            im = gaxes[irow][icol].scatter(data_df1[colname][::stride_plots],data_df1[rowname][::stride_plots], c = data_df1['color'][::stride_plots],
              s = scattersize, cmap = cmap, vmin = vmin, vmax = vmax)
            equalline = np.linspace(-6,2,100)
            gaxes[irow][icol].plot(equalline,equalline,'w:')
            #gaxes[irow][icol].scatter([np.log10(Ashley_opt[icol])],[np.log10(Ashley_opt[irow])],s = 15, c='tab:red')
            # print(parnames)
            gaxes[irow][icol].set_xlim([axislim[icol][0],axislim[icol][1]])
            gaxes[irow][icol].set_ylim([axislim[irow][0],axislim[irow][1]])

          ## density plots
          if icol == irow:
            gaxes[irow][icol].hist(data_df1[rowname],histtype = 'stepfilled', color = sbcolorcyclemuted[4], edgecolor = sbcolorcycledark[4])
            gaxes[irow][icol].set_xlim([axislim[irow][0],axislim[irow][1]])
            ylims = gaxes[irow][icol].get_ylim()
            #gaxes[irow][icol].plot([np.log10(Ashley_opt[icol]),np.log10(Ashley_opt[icol])],ylims,':',color = 'tab:red')

          if icol <= irow:
              if irow==(len(parnames)-1): # bottom row
                gaxes[irow][icol].set_xlabel(colname,fontsize = fontsize)
                gaxes[irow][icol].set_xticks(axis_ticks[icol])
                gaxes[irow][icol].set_xticklabels(tick_labels[icol], fontsize = fontsize)
              else: # not bottom row
                gaxes[irow][icol].set_xticks(axis_ticks[icol])
                gaxes[irow][icol].tick_params(labelbottom=False)
              if icol==0: # leftmost column
                gaxes[irow][icol].set_ylabel(rowname,fontsize = fontsize)
                gaxes[irow][icol].set_yticks(axis_ticks[irow])
                gaxes[irow][icol].set_yticklabels(tick_labels[irow], fontsize = fontsize)
              else:
                gaxes[irow][icol].tick_params(labelleft=False)
                gaxes[irow][icol].set_yticks(axis_ticks[irow])
              if (irow==0 and icol==0): # top left panel fix label
                gaxes[irow][icol].tick_params(labelleft=False)
                gaxes[irow][icol].set_yticks([])
                gaxes[irow][icol].set_yticklabels([])
                gaxes[irow][icol].set_ylabel('')

    # ax_colorbar = fig.add_axes([0.8,0.3,0.03,0.4])
    # cb = plt.colorbar(im,cax = ax_colorbar, ticks = [np.log10(2.5/2.5),np.log10(1.0/2.5),np.log10(1.5/2.5),
    #                                   np.log10(2.0/2.5),np.log10(2.5/2.5),
    #                                   np.log10(3.0/2.5),np.log10(3.5/2.5),
    #                                   np.log10(4.0/2.5),np.log10(4.5/2.5)],
    #                                   extend = 'both')
    # cb.ax.set_yticklabels(['2.5','1.0','1.5','2.0','2.5','3.0','3.5','4.0','4.5'])
    # ax_colorbar.text(-0.8,0.35,'average\n time factor')


    plt.gcf().set_size_inches(figsize, figsize)
    if method == 'ME':
        str1 = 'Full Likelihood $\\mathcal{L}$'
    elif method == 'MF':
        str1 = 'Deterministic Likelihood $\\mathcal{L}^*$'
    if md.model_properties[model]['integrator_name'] == 'Berg':
        str2 = 'Berg model'    
    elif md.model_properties[model]['integrator_name'] == 'WeakStrong':
        str2 = 'Multiple state model' 


    #plt.gcf().text(0.5,0.8,str2+'\n'+str1+'\n'+'$\\gamma = {}$'.format(gamma),fontsize = fontsize)
    #plt.savefig('pargrid_'+str(chain)+'_hill_'+str(ordermethod)+'.png',dpi=300)
    plt.savefig('pargrid_{}_{}_{}_{}_deterministic.png'.format(model,method,gamma,chain),dpi=300,bbox_inches='tight')
    plt.show()
