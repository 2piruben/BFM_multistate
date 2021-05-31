import data_analysis as ed
import numpy as np
from scipy.stats import norm, uniform, multivariate_normal
from scipy.optimize import minimize
import sys,ast
from random import choices,seed,random
from tqdm import tqdm
from p_tqdm import p_umap
from functools import partial
import models as md


gamma = 300 # generic label

## Aproximate Bayesian Computation Sequential Monte Carlo. It parallelizes the search using p_umap

def GeneratePar_Multimodel(processcall = 0, models = [], modelweights = None,
                previouspars = None, previousweights = None,
                eps_dist = 10000, kernels = None, includecondition = 'all'):
# Generates a parameter point from the the list of models "models" each one with a weight "modelweights"
# "previouspars" is used to generate the sampling kernel from the previous iteration if available
# "eps_dist" is the target distance of the SMC step
# processcall is a dummy variable that can be useful when tracking the function performance
# it also allows the use of p_tqdm mapping that forces the use of an iterator 

        seed() # setting random seeds for each thread/process to avoid having the same random sequence in each thread
        np.random.seed()
        evaluated_distances = []
        distance = eps_dist + 1 # original distance is beyond eps_dist

        while distance>eps_dist: # until a suitable parameter is found
            proposed_model = choices(models, weights = modelweights)[0]
            integrator_name = md.model_properties[proposed_model]['integrator_name']
            if 'includecondition' in md.model_properties[proposed_model]:
                includecondition = md.model_properties[proposed_model]['includecondition']


            if previouspars is None: # if is the first iteration
                proposed_pars = md.sampleprior(proposed_model)
                # print('proposed_pars_firsttime', proposed_model,  proposed_pars)  

            else:
                selected_pars = choices(previouspars[proposed_model], weights = previousweights[proposed_model])[0]
                proposed_pars = np.array(selected_pars) + kernels[proposed_model].rvs()

            if md.model_properties[proposed_model]['sto'] == 'ME': # full stochastic integration
                sto_trajs = True
            elif md.model_properties[proposed_model]['sto'] == 'MF':
                sto_trajs = False   

            resurrectionzeros = md.model_properties[proposed_model]['resurrectionzeros']
            # print('proposed_pars', proposed_pars)
            if (md.evaluateprior(proposed_pars,proposed_model) > 0):

                distance = 0

                gammas = [300,500,1300]
                for gamma in gammas:
                    integration_pars = md.prior_to_integrator(proposed_pars,proposed_model,gamma)
                    distance += ed.DistanceLikelihood(gamma, integrator_name, integration_pars,
                                return_trajs = False, sto_trajs = sto_trajs,
                                resurrectionzeros = resurrectionzeros, includecondition = includecondition)
                # print('with distance: {} and eps_dist" {}\n'.format(distance,eps_dist))
                evaluated_distances.append(distance)
            else:
                distance = 2*eps_dist


        # Calculate weight
        if previouspars is None:
            weight = 1
        else:
            sum_denom = 0
            for ipars,pars in enumerate(previouspars[proposed_model]):
                kernel_evaluation = kernels[proposed_model].pdf(proposed_pars-pars)
                sum_denom += kernel_evaluation*previousweights[proposed_model][ipars]

            weight = md.evaluateprior(proposed_pars, model = proposed_model)/sum_denom
        return proposed_pars, distance, weight, evaluated_distances, proposed_model


#### 

def GeneratePars_Multimodel(models = [], modelweights = [], previouspars = None,
                 previousweights = None, eps_dist = 10000, Npars = 1000,
                 previouskerneldict = None, kernelfactor = 1.0):
# Calls GeneratePar in parallel using p_umap to generate Npars parameter valid points
    previouscovardict = {}
    kerneldict = {}

    if previouspars is not None: # Loop to remove model from sampling if does not have points left
        for imodel,model in enumerate(models):
            if not previouspars[model]: # if there are not pars left for this model
                modelweights[imodel] = 0 # do not attempt to sample it

    if previouspars is not None:
        for imodel,model in enumerate(models):
            if len(previouspars[model])>3: # if  
                previouscovardict[model] = 2.0*kernelfactor*np.cov(np.array(previouspars[model]).T)
        # print('covariance matrix previous parset:',previouscovar)
                kerneldict[model] = multivariate_normal(cov = previouscovardict[model], allow_singular = True)

            else: # if there are very few points left, use the last kernel 
                kerneldict[model] = previouskerneldict[model]
            # print('Sample from kernel for model ', model, kerneldict[model].rvs())

    else:
        kerneldict = None # first evaluation, when there is no parameters (or need) to estimate kernel

    trials = 0

    # for N in tqdm(range(Npars)):
    #   GenerateParstePar(0,model = 'Berg',gamma = gamma, previouspars = previouspars,
    #   previousweights = previousweights, eps_dist = eps_dist, kernel = kernel)

    results = p_umap(
        partial(GeneratePar_Multimodel, models = models, modelweights = modelweights,
            previouspars = previouspars, previousweights = previousweights,
            eps_dist = eps_dist, kernels = kerneldict), range(Npars))
    
    newpars = {}
    newweights = {}
    accepted_distances = []
    evaluated_distances = []

    for model in models:
        newpars[model] = []
        newweights[model] = []

    for result in results:
        accepted_distances.append(result[1])    
        evaluated_distances.extend(result[3]) # flatten list
        newpars[result[4]].append(result[0])
        newweights[result[4]].append(result[2])

    for model in models:

        newweights[model] /= np.sum(newweights[model])

    print("acceptance rate:", Npars/len(evaluated_distances))
    print("min accepted distance: ",np.min(accepted_distances))
    print("median accepted distance: ",np.median(accepted_distances))
    print("median evaluated distance: ",np.median(evaluated_distances))

    return(newpars,newweights,accepted_distances, Npars/len(evaluated_distances), kerneldict)


def Sequential_ABC_Multimodel(models = ['BergU','WeakStrongU'],initial_dist = 30000,
                   final_dist =500, Npars = 1000, adaptative_kernel = False):

# Main function call for the SMC.
# - models - is a list of models from models.py
# initial_dist, final_dist - initial and final distance for the SMC
# Npars - number of parameters to sample per iteration
# prior_label - if numeric, it can be used to restart the SMC at an iteration of a previous run
# adaptative_kernel if True allows to change the bandwith of the kernel if the number of points accepted is too large/small

    pars = None # dictionary containing the accepted parameters for each model
    weights = None

    distance = initial_dist
    idistance = 0
    not_converged = True
    last_round = False
    kernelfactor = 1.0
    kerneldict = None

    modelweights = np.ones(len(models))/len(models)

    while not_converged:
        idistance += 1
        print("SMC step with target distance: {}".format(distance))

        pars,weights,accepted_distances,acceptance,kerneldict = GeneratePars_Multimodel(models = models,
            modelweights = modelweights,
            previouspars = pars, previousweights = weights,
            eps_dist = distance, Npars = Npars,
            previouskerneldict = kerneldict, kernelfactor = kernelfactor)
        proposed_dist = np.median(accepted_distances)
        if last_round is True:
            not_converged = False
            label = 'final'
        else:
            label = idistance
        if proposed_dist<final_dist:
            distance = final_dist
            last_round = True
        else:
            distance = proposed_dist

        foldername = ''
        for model in models:
            foldername += model
            foldername += '_'

        for model in models:
            np.savetxt('smc_multimode/{}/pars_{}_{}.out'.format(foldername,model,label), pars[model])
            np.savetxt('smc_multimode/{}/weights_{}_{}.out'.format(foldername,model,label), weights[model])
            np.savetxt('smc_multimode/{}/distances_{}_{}.out'.format(foldername,model,label), accepted_distances)

        if acceptance < 0.1 and kernelfactor>0.1 and adaptative_kernel:
            kernelfactor = kernelfactor * 0.7
            print('Reducing kernel width to : ',kernelfactor)
        elif acceptance > 0.5 and kernelfactor<1 and adaptative_kernel:
            kernelfactor = kernelfactor / 0.7
            print('Increasing kernel width to : ',kernelfactor)


def DeterministicMinimizations(method = 'Nelder-Mead',
                               models = ['WeakStrong_weakcatch','WeakStrong_fullcatchbond','WeakStrongU','BergO'],
                               repeats = 15):
# Search for the maximum of the Likelihood function using a valid method from scipy.optimize.

    for model in models:
        print('Evaluating Model {}'.format(model))
        for repeat in range(repeats):   
            sto = md.model_properties[model]['sto']
            if sto == 'ME': # full stochastic integration
                sto_trajs = True
            elif sto == 'MF':
                sto_trajs = False

            if 'likelihoodcondition' in md.model_properties[model]:
                likelihoodconditions = md.model_properties[model]['likelihoodcondition']
            else:
                likelihoodconditions = 'all'

            distancefunction300 = partial(ed.DistanceLikelihood, gamma = 300, model = md.model_properties[model]['integrator_name'],
                               return_trajs = False, sto_trajs = sto_trajs, resurrectionzeros = md.model_properties[model]['resurrectionzeros'],
                               includecondition = likelihoodconditions)

            distancefunction500 = partial(ed.DistanceLikelihood, gamma = 500, model = md.model_properties[model]['integrator_name'],
                               return_trajs = False, sto_trajs = sto_trajs, resurrectionzeros = md.model_properties[model]['resurrectionzeros'],
                               includecondition = likelihoodconditions)

            distancefunction1300 = partial(ed.DistanceLikelihood, gamma = 1300, model = md.model_properties[model]['integrator_name'],
                               return_trajs = False, sto_trajs = sto_trajs, resurrectionzeros = md.model_properties[model]['resurrectionzeros'],
                               includecondition = likelihoodconditions)

            distancefull = lambda x :  (distancefunction300(params = md.prior_to_integrator(x,model,300)) +
                                       distancefunction500(params = md.prior_to_integrator(x,model,500)) +
                                       distancefunction1300(params = md.prior_to_integrator(x,model,1300)))


            x0 = GeneratePar_Multimodel(processcall = 0, models = [model], modelweights = None,
                        previouspars = None, previousweights = None, eps_dist = 10000, kernels = None)[0]
            # x0 can be replaced with the MAP of the ABC-SMC

            print('Optimization {}: x0 = {}'.format(repeat,x0))
            res = minimize(distancefull, x0 = x0, method = method)
            print('Result: (success = {}) {}'.format(res.success, res.x,))
            print('With distance: {}'.format(distancefull(res.x)))
            print('\n')
        print('\n','-'*10,'\n')


def BayesFactors(models = ['WeakStrong_weakcatch','BergO','WeakStrongU','WeakStrong_fullcatchbond'], threshold = 1000):
# Using Didelot method to calculate BF from summary statistic by approximating
# the evidence for the model as the sum of weights of ABC

    sumweights_list = []
    # for model in models:
    for model in models:
      for bandwidth in [1.0]:
        sto = md.model_properties[model]['sto']
        if sto == 'ME': # full stochastic integration
            sto_trajs = True
        elif sto == 'MF':
            sto_trajs = False
        distancefunction300 = partial(ed.DistanceLikelihood, gamma = 300, model = md.model_properties[model]['integrator_name'],
                           return_trajs = False, sto_trajs = sto_trajs, resurrectionzeros = md.model_properties[model]['resurrectionzeros'])

        distancefunction500 = partial(ed.DistanceLikelihood, gamma = 500, model = md.model_properties[model]['integrator_name'],
                           return_trajs = False, sto_trajs = sto_trajs, resurrectionzeros = md.model_properties[model]['resurrectionzeros'])

        distancefunction1300 = partial(ed.DistanceLikelihood, gamma = 1300, model = md.model_properties[model]['integrator_name'],
                           return_trajs = False, sto_trajs = sto_trajs, resurrectionzeros = md.model_properties[model]['resurrectionzeros'])

        distancefull = lambda x :  (distancefunction300(params = md.prior_to_integrator(x,model,300)) +
                                   distancefunction500(params = md.prior_to_integrator(x,model,500)) +
                                   distancefunction1300(params = md.prior_to_integrator(x,model,1300)))        

        ##### Looking for best file to compare
        nn = 10
        low = True
        dis = np.loadtxt('smc/distances_{}_MF_300_{}.out'.format(model,nn))
        print('Testing file {} with max dist: {}'.format(nn,max(dis)))
        while (low is True):
            try:
                dis = np.loadtxt('smc/distances_{}_MF_300_{}.out'.format(model,nn))
                if max(dis)>threshold:
                    print('Testing file {} with max dist: {}'.format(nn,max(dis)))
                    nn += 1
                else:
                    print('Accepted file {} with max dist: {}'.format(nn,max(dis)))
                    nn += 1
                    low = False
            except: # if the file is not found
                print('File note found for model {} with threshold {}'.format(model,threshold))
                dis = np.loadtxt('smc/distances_{}_MF_300_final.out'.format(model))
                print('Using final file with max dist {}'.format(max(dis)))
                nn = 'final'
                low = False
        data = np.loadtxt('smc/pars_{}_MF_300_{}.out'.format(model,nn))
        covar = np.cov(data.T)
        kernel = multivariate_normal(cov = bandwidth*covar)
        perturbations = kernel.rvs(size = len(data))
        newdata = data + perturbations
        print('covar dim for model {} is {}'.format(model,covar.shape))
        sumweights = 0
        for point in tqdm(newdata):
            priorevaluation = md.evaluateprior(point,model)
            #if (priorevaluation > 0):
            if True:
                distance = distancefull(point)
                if distance < threshold:
                    sum_denom = 0
                    for ipars,pars in enumerate(data):
                        kernel_evaluation = kernel.pdf(point-pars)
                        sum_denom += kernel_evaluation
                    sumweights += priorevaluation/sum_denom
        print('Sum of weights for model {} with bd {} is :{}'.format(model, bandwidth, sumweights))
        sumweights_list.append(sumweights)
    
    for i in range(len(models)):
        for j in range (len(models)):
            if i<j:
                print("Approxiamte Bayes factor of {} over {} is {}".format(
                    models[i],models[j], sumweights_list[i]/sumweights_list[j]))




            




    






