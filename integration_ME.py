import numpy as np 
from scipy import linalg
from scipy.integrate import solve_ivp
from scipy import stats
from scipy import sparse
import scipy.sparse.linalg as sp_linalg
import random

Nmax = 13 ## Maximum number of stators allowed by default

def GetEigenElements(M):
# Takes Matrix M and computes the eigendecomposition and the change of basis matrix. This is useful so it does not have to be computed
# every single time the function is integrated.
	ls, vs = linalg.eig(M) # diagonalization of matrix
	vsinv = linalg.inv(vs) # change of basis matrix
	return ls, vs, vsinv

def GetTransitionMatrix(M,DeltaT, eigen_given = False): 
#Given the transition matrix M and a step Delta T, return a matrix of the probability of transition from a 
# state x to a state y in an interval DeltaT

	if eigen_given:
		T = np.zeros_like(M[1])
	else:
		T = np.zeros_like(M)
	Tdim = len(T[:,])
	for inn,n in enumerate(T[:,]):
		v_aux = np.zeros(Tdim)
		v_aux[inn] = 1
		T[inn,:] = Integrate(M,v_aux,[DeltaT], eigen_given = eigen_given)
	return T 

def Integrate(Mext,P0,times,eigen_given = False):
# Given an initial probability vector P0, return the probability vectors Pt at times t
# given a linear rate matrix dP/dt = M P.
# Note that the integration process required to compute the eigendecomposition of the matrix
# to avoid to repeat this each call of Integrate, Mext can include this info so
# if eigen_given = False, Mext is the evolution matrix M
# if eigen_fiven = True, Mext = [ls,vs,inv(vs)]

	if (eigen_given):
		ls,vs,invvs = Mext
	else:
		ls, vs = linalg.eig(Mext) # diagonalization of matrix
		invvs = linalg.inv(vs) 
	a = np.dot(invvs,P0) # decomposition of the initial condition in the eigenbase
	
	# for each timepoint (row) calculate Pt from the exponential components of the eigenvector
	Pt = np.real(np.dot(a * np.exp(np.tensordot(times,ls,0)), vs.T)) 
	return Pt

# Convolution Matrix
# 
convo = np.zeros((Nmax+1,Nmax+1))
n_means = np.arange(Nmax+1,dtype = np.float)
n_means[0] = 0.9
n_means[-1] = Nmax-0.1
concentration = 100

for ncolumn,n in enumerate(n_means):
	convo[:,ncolumn] = stats.betabinom.pmf(np.arange(Nmax+1),Nmax,concentration,concentration*(Nmax/n-1))
	# print('ncolumn', n ,convo[:,ncolumn] )

def Convolute(P):
# Given a matrix of probabilities in time, introduce an inaccuracy of the observation by 
# applying a running kernel along the number of stators e.g.
	return (np.dot(convo,P.T)).T

def Equilibrium(Mext,eigen_given = False):
# Given a evolution matrix return the steady state population. Since they are rate matrices, if all the rates are >0,
# there should be a unique eigenvalue identically zero that corresponds with the steady state dP/dt = LP = 0 -> lambda = 0

	if (eigen_given):
		ls,vs,invvs = Mext
	else:
		ls, vs = linalg.eig(Mext) # diagonalization of matrix
	zero_eigenvalue_pos = np.argmin(np.abs(ls))
	equilibrium_eigenvector = np.real(vs[:,zero_eigenvalue_pos]) # imaginary part should be zero, numerically can give some small imaginary part
	normalized_P = equilibrium_eigenvector/np.sum(equilibrium_eigenvector)
	return normalized_P


def Berg_Matrix(k0, alpha, gamma, N=13):
# Matrix for the speed-rate model so P(n,t) = L P(n,t)
	# Number of sites is N, so matrix must be (N+1,N+1) to accommodate state 0 (redundant but clearer)
	L = np.zeros((N+1,N+1))
	idxs = np.arange(0,N) # this vector will be used frequently for filling the matrix
	# For this model the matrix is tridiagonal. We will fill the matrix diagonal per diagonal
	# First diagonal contains off-rates
	diagonal_top = tuple(np.vstack((idxs,idxs+1)))
	L[diagonal_top] = gamma*k0*(1-np.exp(-alpha/(idxs+1)))*(idxs+1)
	#Bottom diagonal contains on-rates
	diagonal_bottom = tuple(np.vstack((idxs[1:]+1,idxs[1:]))) # [1:] because first element will be introduced separately
	L[diagonal_bottom] = k0*(1-np.exp(-alpha/(idxs[1:])))*(N-idxs[1:])
	L[1,0] = k0*N # introduced separately to avoid numeric error at exp(-alpha/N) when N->0 
	# Middle diagonal
	diagonal_middle = tuple(np.vstack((idxs[1:],idxs[1:]))) # [1:-1] because first and last element will be introduced separately
	L[diagonal_middle] = -k0*((N-idxs[1:])+gamma*idxs[1:])*(1-np.exp(-alpha/(idxs[1:])))
	L[0,0] = -k0*N
	L[-1,-1] = -k0*N*gamma*(1-np.exp(-alpha/N))
	return L 

def WeakStrong_Matrix(kuw, kwu, kws, ksw, N=13):
# Return rate Matrix for the two-state model so P(w,s,t) = L P(w,s,t)
# if we define the dimensions of the matrix as ((N+1)*(N+1))^2, then it can be very slow, but only half of this states
# are available since n+m <= N. So we will redifine the index of the vector with k such that 
# k = (w=0,s=0),(w=1,s=0),(w=0,s=1),(w=2,s=0),(w=1,s=1),(w=0,s=2), .... , (w=1,s=11),(w=0,s=13)
# i.e we have 1 state with zero stators, 2 sates with 1 stators, .... 13 states with 13 stators.
# In total we have (N+1)*(N+2)/2 stators
# we require a function to make the transformation k <-> (n,m) back and forward. 
# These are defined after the matrix   

	M = int((N+2)*(N+1)/2)
	#L = sparse.csr_matrix((M,M))
	L = np.zeros((M,M))
	# print("vector size, ",M)
	# Matrix is filled case by case
	for w in range(N+1):
		for s in range(0,N+1-w):
			# print('Number of stators: ',w,s)
			# Variation of index corresponding to (w,s)
			# for each channel we will have a positive and negative contribution
			index_var = WeakStrong_valuestoindex(w,s,N)
			# Stator recruitment
			if w>0:
				index_temp = WeakStrong_valuestoindex(w-1,s,N)
				L[index_var,index_temp] += kuw * (N-w-s+1)
				#print(index_var,index_temp)
			if w<N:
				L[index_var,index_var] -= kuw * (N-w-s)
			# Stator release
			if (N-w-s>0):
				index_temp = WeakStrong_valuestoindex(w+1,s,N)
				L[index_var,index_temp] += kwu * (w+1)
				#print(index_var,index_temp)
			if w>0:
				L[index_var,index_var] -= kwu * w
			# Weak->Strong stator
			if (s>0):
				index_temp = WeakStrong_valuestoindex(w+1,s-1,N)
				L[index_var,index_temp]  += kws * (w+1)
				# print(index_var,index_temp)
			if (w>0):
				L[index_var,index_var]  -= kws * w
			# Strong->Weak activator
			if (w>0):
				index_temp = WeakStrong_valuestoindex(w-1,s+1,N)
				L[index_var,index_temp]  += ksw * (s+1)
				# print(index_var,index_temp)
			if (s>0):
				L[index_var,index_var]  -= ksw * s		
	return L 


def WeakStrong_indextovalues(j,N=13):
# Given an index return the pair (weak bound, strong bound) to build the WeakStrong transition matrix
# This function could be probably simplified using aritmetic series and modulo operations to vectorize it
	totalstators = 0
	while(j>totalstators):
		totalstators +=1
		j -= totalstators
	return (totalstators - j,j)

def WeakStrong_valuestoindex(w,s,N=13):
# Given a pair (weak bound, strong bound) return the index to build the WeakStrong transition matrix
	T = w + s
	return  int(s + T*(T+1)/2)

def Pnfrom_WS(P,N=13): # given a vector of indexes <w,s> return the probability vector of having n stators bound
	Q = np.zeros(N+1) # output vector
	j = 0
	k = 0
	for k in range(N+1):
		j += k
		Q[k] = np.sum(P[j:j+k+1])
	return Q

def ConditionPwstoN(Peq,Nstar):
	# Given a vector of probabilities P(w,s) return the normalized vector P(w,s|w+s = Nstar)
	# and the sum of the filtered elements before normalization

	# since the vector Pws is ordered, we can slice directly the section where w+s = Nstar
	#sl = slice(Nstar*(Nstar+1)//2 , (Nstar+1)*(Nstar+2)//2) # slice of indexed where w+s = Nstar
	Peq[:Nstar*(Nstar+1)//2] = 0
	Peq[(Nstar+1)*(Nstar+2)//2:] = 0
	sumels = Peq[Nstar*(Nstar+1)//2:(Nstar+1)*(Nstar+2)//2].sum()
	if sumels>0:
		Peq[Nstar*(Nstar+1)//2:(Nstar+1)*(Nstar+2)//2] = Peq[Nstar*(Nstar+1)//2:(Nstar+1)*(Nstar+2)//2]/sumels
		return sumels
	else:
		return -1

def FlowMeanFieldBerg(t,n, k0, alpha, gamma, Nmax=Nmax):
# This is the mean field using f(<N>) = <f(N)>	
	if n==0:
		kon = k0
	else:
		kon = k0 * (1-np.exp(-alpha/n))
	koff = kon*gamma
	return kon*(Nmax - n ) - koff*n


def TrajMeanFieldBerg(k0, alpha, gamma, N0, times, Nmax=Nmax):

	return solve_ivp(FlowMeanFieldBerg, [0,times[-1]], [N0], t_eval = times, args = (k0,alpha,gamma,Nmax)).y[0]
	

def FlowMeanFieldWeakStrong(t,ws, kuw, kwu, kws, ksw, Nmax = Nmax):

	wdot = ((Nmax-ws[0]-ws[1])*kuw - (kwu + kws)*ws[0] + ksw*ws[1])
	sdot = (- (ksw)*ws[1] + kws*ws[0])

	return([wdot,sdot])


def MeanSteadyStateWeakStrong(kuw, kwu, kws, ksw):
		denom = kuw*kws + kuw*ksw + ksw*kwu
		w_inf = Nmax*ksw*kuw / denom
		s_inf = Nmax*kws*kuw / denom
		return w_inf,s_inf

def TrajMeanFieldWeakStrong(kuw, kwu, kws, ksw, w0, s0, times, Nmax=Nmax, method = 'analytical'):
# Analytical uses the analytical expression for <s> and <w> obtained by diagonalizing the system d<w>/dt and d<s>/dt
	if method == 'analytical':
		denom = kuw*kws + kuw*ksw + ksw*kwu
		w_inf = Nmax*ksw*kuw / denom
		s_inf = Nmax*kws*kuw / denom
		K = kuw + kwu + kws + ksw
		lM = -K/2*(1-np.sqrt(1-4*denom/K/K))
		lm = -K/2*(1+np.sqrt(1-4*denom/K/K))
		aM = (lM + kuw + kwu + kws)/(ksw - kuw)
		am = (lm + kuw + kwu + kws)/(ksw - kuw)
		Cm = (ksw - kuw)*(s0 - s_inf - aM*(w0 - w_inf))/(lm - lM)
		CM = w0 - w_inf - Cm
		wt = CM*np.exp(lM*times) + Cm*np.exp(lm*times) + w_inf
		st = CM*aM*np.exp(lM*times) + Cm*am*np.exp(lm*times) + s_inf
		return [wt,st]
# Numerical integration of d<w>/dt and d<s>/dt
	elif method == 'integration':	
		return solve_ivp(FlowMeanFieldWeakStrong, [0,times[-1]], [w0,s0], t_eval = times, args = (kuw, kwu, kws, ksw,Nmax)).y



def stats(P,integrator = 'Berg', N = 13, weakstrong_output = False): # given a time array of Probability vectors as returned by Integrate, return the mean and variance in time
 	if integrator == 'Berg':
 # if weakstrong_output is True also return the expected weak and strongly bound stators 
 		
 		# print('sizes: ', np.shape(P), np.shape(np.arange(N+1)),'\n')
 		mean = np.dot(P,np.arange(N+1))
 		mean2 = np.dot(P,np.arange(N+1)*np.arange(N+1))

 		# print('means',mean,mean2)
 		std = np.sqrt(abs(mean2-mean*mean))# abs just dodges some values close to zero that can become numerically negative
 		return mean,std
 	elif integrator == 'WeakStrong':
 		ws = [np.sum(WeakStrong_indextovalues(j)) for j in range(np.shape(P)[-1])]
 		mean = np.dot(P,np.array(ws))
 		mean2 = np.dot(P,np.array(ws)*np.array(ws))
 		std = np.sqrt(abs(mean2-mean*mean))# abs just dodges some values close to zero that can become numerically negative
 		if weakstrong_output is False:
 			return mean,std
 		else:
 			w = [WeakStrong_indextovalues(j)[0] for j in range(np.shape(P)[-1])]
 			s = [WeakStrong_indextovalues(j)[1] for j in range(np.shape(P)[-1])]
 			mean_weak = np.dot(P,np.array(w))
 			mean_strong = np.dot(P,np.array(s))
 			return mean,std,mean_weak,mean_strong 


 	else:
 		print('Unknown model for stats!!!!')
 	

# 	return np.dot(P,np.arange(len(P)))



def PwstoPn(Pws, Nmax = 13): 
# given a multidimensional Pws (times x states) vector for the weakstrongmodel, returns a multidimensional (times x stators)

	time_dim = Pws.shape[0]
	Pn = np.zeros((time_dim,Nmax+1)) # initializing return matrix

	n_dummy = 0 # last index of the matrix evaluated
	for n_stators in range(Nmax):
		Pn[:,n_stators] = np.sum(Pws[:,n_dummy:n_dummy+n_stators+1], axis = 1)
		n_dummy += n_stators + 1
	return Pn

	
def Normalize(P,Ptol = 1E-8):
# Normalize a probability vector by removing negative values
# and setting smallest values to Ptol to avoid
	P = np.abs(P) + Ptol
	P = P/np.sum(P)
	return P

def Get_Berg_Trajectory(N0, k0, alpha, gamma, times):
# Run Gillespie trajectory for Bergs model	

	time = times[0]
	itimecounter = 1
	nextrecordtime = times[itimecounter]
	timelength = len(times)
	N = N0
	oldN = N
	reactiontimes = [0]	
	Ns = np.zeros_like(times)
	Ns[0] = N0

	# print("Calculating trajectory for pars: {}".format((k0,alpha,gamma)))

	while time<times[-1]:

		r1 = random.uniform(0,1)
		r2 = random.uniform(0,1)

		if (N>0 and N<Nmax):

			prop_on = (Nmax - N)*k0*(1 - np.exp(-alpha/N))
			prop_off = N*gamma*k0*(1 - np.exp(-alpha/N))

			sumprop = prop_on + prop_off
			threshold = prop_on/(prop_off+prop_on)

			time -= 1.0/sumprop*np.log(r1)

			if r2>threshold:
				oldN = N
				N = N - 1

			else:
				oldN = N
				N = N + 1

		elif N==0:

			prop =  Nmax*k0
			time  -= 1.0/prop*np.log(r1)	
			oldN = 0
			N = 1

		elif N==Nmax:

			prop = Nmax*gamma*k0
			time -= 1.0/prop*np.log(r1)	
			oldN = Nmax
			N = Nmax - 1

		# print('at time {} the occupancy is {}'.format(time,N))
		# print('Next reaction time: ',nexttime)
		while ((time>nextrecordtime) and (itimecounter<timelength-1)):
			# print('Updating time :', nextrecordtime)
			Ns[itimecounter]= oldN
			itimecounter += 1
			nextrecordtime = times[itimecounter]

	Ns[-1] = oldN

	return Ns
			
def Get_WeakStrong_Trajectory(W0, S0, kuw, kwu, kws, ksw, times, ceiling_N = 1000, return_time = False):

# Run Gillespie trajectory for Bergs model	
# ceiling_N sets a ceiling of occupancy to stop the simulations

	# print('Starting Gillespie traj at position', W0, S0)
	time = times[0]
	itimecounter = 1
	nextrecordtime = times[itimecounter]
	timelength = len(times)
	X = np.array([W0,S0]) # the variables are stored in a list X
	oldX = X
	reactiontimes = [0]	
	Xs = np.zeros((len(times),2))
	Xs[0,:] = X
	propensities = np.array([0.0,0.0,0.0,0.0]) # order will be kuw,kwu,kws,ksw
	stoichiometries = np.array([[1,0],[-1,0],[-1,1],[1,-1]])

	# print("Calculating trajectory for pars: {}".format((k0,alpha,gamma)))

	while ( (time<times[-1]) and (oldX[0]+oldX[1])<ceiling_N):

		r1 = random.uniform(0,1)
		r2 = random.uniform(0,1)

		# print('Inside Gillespie loop')
		N = X[0]+X[1]
		propensities[0] = (Nmax-N)*kuw 
		propensities[1] = X[0]*kwu
		propensities[2] = X[0]*kws
		propensities[3] = X[1]*ksw

		# print('Parameters:',kuw,kwu,kws,ksw)
		# print('Propensities:',propensities)

		sumprop = propensities.sum()
		time -= 1.0/sumprop*np.log(r1)

		probabilities = np.cumsum(propensities/sumprop) # CDF
		selected_reaction = np.where(probabilities>r2)[0][0]

		oldX = X

		# print('Selected stoichiometry:', stoichiometries[selected_reaction])		
		X = X + stoichiometries[selected_reaction]

		while ((time>nextrecordtime) and (itimecounter<timelength-1)):
			# print('Updating time :', nextrecordtime)
			Xs[itimecounter,:]= oldX
			itimecounter += 1
			nextrecordtime = times[itimecounter]

		Xs[-1,:] = oldX

	if return_time:
		return Xs, time 
	else:
		return Xs


def Get_WeakStrong_MFPT(W0, S0, NF, kuw, kwu, kws, ksw, N = 100):

	times = []
	times_integration = np.linspace(0,1000)
	for n in range(N):
		Xs,time = Get_WeakStrong_Trajectory(W0, S0, kuw, kwu, kws, ksw, times_integration, ceiling_N = NF, return_time = True)
		times.append(time)
	return times


