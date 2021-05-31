import numpy as np
from scipy.stats import norm, uniform, multivariate_normal

### Models used in the analysis including properties of sampling 
# 'integrator_name' : 'Berg' or 'WeakStrong' depending if they derive from the speed-rate of the multistate model
# 'sto': 'MF' or 'ME' to determine if a stochastic trajectory is required.
# 'resurrectionzeros': determine the alignment at resurrection (as described in data_analysis.py)
# 'lower_limits', 'scale_limits': limits for the prior distributions
# 'parnames': meaning of each parameter with a label that can be printed
# 'fontsize', 'scattersize': properties for plotting
# 'gamma_file': default label for outputfiles (can be ignored)

model_properties = { # dict of dicts containing the equivalence between metamodel names and their properties 
# use for hierarchical model comparison
    'BergU':{ # speed-rate model where alpga and zeta are allow to depend freely with the load
              'integrator_name': 'Berg',
              'sto' : 'MF',
              'resurrectionzeros' : 'withzeros',
              'lower_limits' : np.array([-2.5,-3.0,-1.5,-2.5,-1.5,-2.5,-1.5]),
              'scale_limits' : np.array([3.0,3.0,3.0,3.0,3.0,3.0,3.0]),
              'parnames' : ['$\\log_{10} k_0$',
                           '$\\log_{10}\\alpha_{300}$','$\\zeta_{300}$',
                           '$\\log_{10}\\alpha_{500}$','$\\zeta_{500}$',
                           '$\\log_{10}\\alpha_{1300}$','$\\zeta_{1300}$'],
              'fontsize': 16,
              'scattersize': 5,
              'gamma_file': 300
            },

    'BergO':{ # original speed-rate model used (the one used in the manuscript)
              'integrator_name': 'Berg',
              'sto' : 'MF',
              'resurrectionzeros' : 'withzeros',
              'lower_limits' : np.array([-3.5,1.0,-1.0,-1.0,-2.0]), 
              'scale_limits' : np.array([3.0,3.0,3.0,3.0,3.0]),
              'parnames': ['$\\log_{10} k_0$','$\\log_{10} \\kappa$',
                           '$\\zeta_{300}$','$\\zeta_{500}$','$\\zeta_{1300}$'],
              'gamma_file': 300
            },

    'WeakStrongU':{# general two-state model
              'integrator_name': 'WeakStrong',
              'sto' : 'MF',
              'resurrectionzeros' : 'withzeros',
              'stallcondition' : 'nounbinding',
              'lower_limits' : np.array([-4,-3.0,-6.0,-5.0,-4,-4.0,-5.0,-5.0,-4,-4.0,-5.0,-5.0]),
              'scale_limits' : np.array([3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0]),
              'parnames': ['$\\log_{10} k_{uw,300}$','$\\log_{10} k_{wu}^{300}$', '$\\log_{10} k_{ws}^{300}$', '$\\log_{10} k_{sw}^{300}$',
                '$\\log_{10} k_{uw}^{500}$', '$\\log_{10} k_{wu}^{500}$', '$\\log_{10} k_{ws}^{500}$', '$\\log_{10} k_{sw}^{500}$',
                '$\\log_{10} k_{uw}^{1300}$', '$\\log_{10} k_{wu}^{1300}$', '$\\log_{10} k_{ws}^{1300}$', '$\\log_{10} k_{sw}^{1300}$'],
              'gamma_file': 300
            },

    'HillLangmuir':{
              'integrator_name' : 'WeakStrong',
              'sto' : 'MF',
              'resurrectionzeros' : 'withzeros',
              'stallcondition' : 'nounbinding',
              'lower_limits' : np.array([-4,-4.0,-4.0,-4.0,-4.0,-4.0]),
              'scale_limits' : np.array([3.0,3.0,3.0,3.0,3.0,3.0]),
              'parnames' : ['$\\log_{10} k_{on}^{300}}$','$\\log_{10} k_{off}^{300}}$',
                            '$\\log_{10} k_{on}^{500}}$','$\\log_{10} k_{off}^{500}}$',
                            '$\\log_{10} k_{on}^{1300}}$','$\\log_{10} k_{off}^{1300}}$'],
              'gamma_file': 300
    },

    'HillLangmuir_release':{
              'integrator_name' : 'WeakStrong',
              'sto' : 'MF',
              'resurrectionzeros' : 'withzeros',
              'stallcondition' : 'nounbinding',
              'lower_limits' : np.array([-4,-4.0,-4.0,-4.0,-4.0,-4.0]),
              'scale_limits' : np.array([3.0,3.0,3.0,3.0,3.0,3.0]),
              'parnames' : ['$\\log_{10} k_{on}^{300}}$','$\\log_{10} k_{off}^{300}}$',
                            '$\\log_{10} k_{on}^{500}}$','$\\log_{10} k_{off}^{500}}$',
                            '$\\log_{10} k_{on}^{1300}}$','$\\log_{10} k_{off}^{1300}}$'],
              'gamma_file': 300,
              'likelihoodcondition': ['release']
    },
    'HillLangmuir_resurrection':{
              'integrator_name' : 'WeakStrong',
              'sto' : 'MF',
              'resurrectionzeros' : 'withzeros',
              'stallcondition' : 'nounbinding',
              'lower_limits' : np.array([-4,-4.0,-4.0,-4.0,-4.0,-4.0]),
              'scale_limits' : np.array([3.0,3.0,3.0,3.0,3.0,3.0]),
              'parnames' : ['$\\log_{10} k_{on}^{300}}$','$\\log_{10} k_{off}^{300}}$',
                            '$\\log_{10} k_{on}^{500}}$','$\\log_{10} k_{off}^{500}}$',
                            '$\\log_{10} k_{on}^{1300}}$','$\\log_{10} k_{off}^{1300}}$'],
              'gamma_file': 300,
              'likelihoodcondition': ['resurrection']
    },
    'WeakStrong_fullcatchbond':{
              'integrator_name' : 'WeakStrong',
              'sto' : 'MF',
              'resurrectionzeros' : 'withzeros',
              'stallcondition' : 'nounbinding',
              'lower_limits' : np.array([-4,-3.0,-6.0,-5.0,-4.0,-5.0,-5.0,-4.0,-5.0,-5.0]),
              'scale_limits' : np.array([3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0]),
              'parnames' : ['$\\log_{10} k_{uw}$', '$\\log_{10} k_{wu}^{300}$','$\\log_{10} k_{ws}^{300}$','$\\log_{10} k_{sw}^{300}$',
                           '$\\log_{10} k_{wu}^{500}$','$\\log_{10} k_{ws}^{500}$','$\\log_{10} k_{sw}^{500}$',
                           '$\\log_{10} k_{wu}^{1300}$','$\\log_{10} k_{ws}^{1300}$','$\\log_{10} k_{sw}^{1300}$'],
              'gamma_file': 300
    },
    'WeakStrong_weakcatch':{
              'integrator_name' : 'WeakStrong',
              'sto' : 'MF',
              'resurrectionzeros' : 'withzeros',
              'stallcondition' : 'nounbinding',
              'lower_limits' : np.array([-4,-5.0,-3.0,-6.0,-4.0,-5.0,-4.0,-5.0]),
              'scale_limits' : np.array([3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0]),
              'parnames' : ['$\\log_{10} k_{uw}$','$\\log_{10} k_{sw}$',
                            '$\\log_{10} k_{wu}^{300}$','$\\log_{10} k_{ws}^{300}$',
                            '$\\log_{10} k_{wu}^{500}$','$\\log_{10} k_{ws}^{500}$',
                            '$\\log_{10} k_{wu}^{1300}$','$\\log_{10} k_{ws}^{1300}$'],
              'gamma_file': 300
    }
    }


def sampleprior(model = None):
# given a model from model_properties return a point sampled from the prior distribution
    prior = []
    for il,low in enumerate(model_properties[model]['lower_limits']):
        prior.append(uniform.rvs(loc = model_properties[model]['lower_limits'][il],
                                 scale = model_properties[model]['scale_limits'][il]))
    return prior

def evaluateprior(pars, model = None):
# given a point, return the value of the prior probability distribution at that point
    prior = 1
    for il,low in enumerate(model_properties[model]['lower_limits']):
            prior *= uniform.pdf(pars[il],loc = model_properties[model]['lower_limits'][il],
                scale = model_properties[model]['scale_limits'][il])
    return prior

def prior_to_integrator(pars, model = 'Berg', gamma = 300):
# transform the vector of candidate parameteres (point of the ABC) into the canonical parameters (biohpysical parameters of the model)
# used by the corresponding integrator

    if model == 'BergU':
        gammadict = {300:0,500:2,1300:4}
        igamma = gammadict[gamma]
        real_pars = [10**pars[0],10**pars[igamma+1],np.exp(pars[igamma+2])]
    elif model == 'BergO':
        gamma_in_friction_units = {300:0.210, 500:0.825, 1300:13.675}
        torque_per_stator= {300:27.2, 500:53.1, 1300:177.9}
        gammadict = {300:0, 500:1, 1300:2}
        igamma = gammadict[gamma]
        speed_per_stator = torque_per_stator[gamma]/gamma_in_friction_units[gamma] 
        real_pars = [10**pars[0],10**(pars[1])/speed_per_stator,np.exp(pars[2+igamma])]
    elif model == 'WeakStrongU':
        gammadict = {300:0,500:4,1300:8}
        igamma = gammadict[gamma]
        real_pars = [10**pars[0+igamma],10**pars[1+igamma],
                     10**(pars[2+igamma]),10**(pars[3+igamma])]                  
    elif model == 'HillLangmuir':
        gammadict = {300:0,500:2,1300:4}
        igamma = gammadict[gamma]
        real_pars = [10**pars[igamma],10**pars[igamma+1],1E-10,1E5]
    elif model == 'HillLangmuir_release':
        gammadict = {300:0,500:2,1300:4}
        igamma = gammadict[gamma]
        real_pars = [10**pars[igamma],10**pars[igamma+1],1E-10,1E5]
    elif model == 'HillLangmuir_resurrection':
        gammadict = {300:0,500:2,1300:4}
        igamma = gammadict[gamma]
        real_pars = [10**pars[igamma],10**pars[igamma+1],1E-10,1E5]
    elif model == 'WeakStrong_fullcatchbond':
        gammadict = {300:0,500:3,1300:6}
        igamma = gammadict[gamma]
        real_pars = [10**pars[0],10**pars[igamma+1],10**pars[igamma+2],10**pars[igamma+3]]
    elif model == 'WeakStrong_weakcatch':
        gammadict = {300:0,500:2,1300:4}
        igamma = gammadict[gamma]
        real_pars = [10**pars[0],10**pars[igamma+2],10**pars[igamma+3],10**pars[1]]
    return real_pars




