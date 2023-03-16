import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import linalg
import scipy.stats as sts 
from matplotlib.gridspec import GridSpec
import seaborn as sns
import warnings
import matplotlib.cbook
import torch

from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

Ni = 0
Mi = 1
Wi = 2
IAi = 3
IBi = 4
IctxAi = 5
IctxBi = 6

## simulation constants

dt=0.01
T_DUR = 7
time = np.arange(0,T_DUR,dt)

STIM_beg = int(2 // dt)
STIM_end = int(T_DUR // dt)

CTX_beg = int(0 // dt)
CTX_end = int(T_DUR  // dt) +1 


trials = 50

### BEST SIGMAS
# Yves: Mante 3-populations + 1 population. Third population is swaped
# by Beiran's 3 fixed point network w/ 3 populations
# n1 (0), m1 (1), w (2), IA (3), IB (4), IctxA (5), IctxB (6)

Yves_Sigma1 = np.eye(7,7)
Yves_Sigma1[Ni,IBi] = Yves_Sigma1[IBi,Ni] = -87
Yves_Sigma1[Ni,IctxBi] = Yves_Sigma1[IctxBi,Ni] =-52
Yves_Sigma1[IctxAi,IctxAi] = 1000
Yves_Sigma1[IctxBi,IctxBi] = 8
Yves_Sigma1[IBi,IBi] = 3

Yves_Sigma2 =  np.eye(7,7)
Yves_Sigma2[Ni,IAi] = Yves_Sigma2[IAi,Ni] = -87
Yves_Sigma2[Ni,IctxAi] = Yves_Sigma2[IctxAi,Ni] =52
Yves_Sigma2[IctxAi,IctxAi] = 8
Yves_Sigma2[IctxBi,IctxBi] = 1000
Yves_Sigma2[IAi,IAi] = 3


Yves_Sigma3 = np.zeros([7,7])
Yves_Sigma3[Mi,Ni] = Yves_Sigma3[Ni,Mi] = -270
Yves_Sigma3[Mi,Mi] = 164


Sigmas_A1 = np.array([Yves_Sigma1, Yves_Sigma2,Yves_Sigma3])
Sigmas_A1[:,Ni,Ni] =5000
Sigmas_A1[:,Mi,IAi] = Sigmas_A1[:,IAi,Mi] = 0
Sigmas_A1[:,Mi,IBi] = Sigmas_A1[:,IBi,Mi] = 0
Sigmas_A1[:,Mi,IctxAi] = Sigmas_A1[:,IctxAi,Mi] = 0
Sigmas_A1[:,Mi,IctxBi] = Sigmas_A1[:,IctxBi,Mi] = 0


# PFC

# n1p (0), m1p (1),  Ip_ctxA (2), Ip_ctxB (3)

PFC_Sigma = np.eye(4,4)
PFC_Sigma[Ni,Mi] = PFC_Sigma[Mi,Ni] =1.6
PFC_Sigma[Ni,2] = PFC_Sigma[2,Ni] =4
PFC_Sigma[Ni,3] = PFC_Sigma[3,Ni] =-4
PFC_Sigma[Ni,Ni] =500



## generate network helpers


def gram_schmidt(vecs):
    vecs_orth = []
    vecs_orth.append(vecs[0] / np.linalg.norm(vecs[0]))
    for i in range(1, len(vecs)):
        v = vecs[i]
        for j in range(i):
            v = v - (v @ vecs_orth[j]) * vecs_orth[j]
        v = v / np.linalg.norm(v)
        vecs_orth.append(v)
    return np.array(vecs_orth)

def gram_schmidt_1(v,vecs):

    for i in range(len(vecs)):
        v1 = vecs[i] / np.linalg.norm(vecs[i])
        v = v - (v @ v1) * v1
    return v

def gram_factorization(G):
    """
    The rows of the returned matrix are the basis vectors whose Gramian matrix is G
    :param G: ndarray representing a symmetric semidefinite positive matrix
    :return: ndarray
    """
    #return la.cholesky(G)
    w, v = np.linalg.eigh(G)
    if any(w<0):
      print('not PSD, truncated negative evals to 0, %f' % min(w))
      w[w<0] = 0
    x = v * np.sqrt(w)
    return x

def generate_network_sigmas(Sigmas, N):

    n_pop = len(Sigmas)
    As = [gram_factorization(S) for S in Sigmas]

    pops = []

    for ni in range(n_pop):
      X = np.random.randn(len(Sigmas[0]),N//n_pop)
      X = gram_schmidt(X)
      X=(X.T / np.std(X,1)).T # following Manuel's code
      pop = As[ni] @ X
      pops.append(pop)

    return np.concatenate(pops,1)

## 

gaussian_norm = (1/np.sqrt(np.pi))
gauss_points, gauss_weights = np.polynomial.hermite.hermgauss(200)
gauss_points = gauss_points*np.sqrt(2)

def phi (mu, delta0):
    integrand = np.tanh(mu+np.sqrt(delta0)*gauss_points)
    return gaussian_norm * np.dot (integrand,gauss_weights)
    
def phi_prime (mu, delta0):
    integrand = 1 - (np.tanh(mu+np.sqrt(delta0)*gauss_points))**2
    return gaussian_norm * np.dot (integrand,gauss_weights)

def phi_prime_num (mu, delta0,N=10000):
    zs = np.random.randn(N)
    sol = np.mean(1 - (np.tanh(mu+np.sqrt(delta0)*zs))**2)
    return sol
    
def _phi_prime(mu, delta0):
    sol = phi_prime(mu, delta0)
    return sol
    if delta0 > 50: 
        sol = phi_prime_num(mu, delta0)
    else:
        sol = phi_prime(mu, delta0)
    return sol
        
def generate_stimuli(time, noise=0.1, ma=np.nan, mb=np.nan, ctxA=np.nan):
  
    if np.isnan(ma):
        ma = np.random.uniform(-1,1)
        ma = np.random.choice([-1,1])
    if np.isnan(mb):
        mb = np.random.uniform(-1,1)
        mb = np.random.choice([-1,1])
    if np.isnan(ctxA):
        ctxA = np.random.choice([0,1])

    u_A = np.ones_like(time) * ma
    u_A += np.random.normal(0,noise,len(time))
    u_A[:STIM_beg] = u_A[STIM_end:] = 0

    u_B = np.ones_like(time) * mb
    u_B += np.random.normal(0,noise,len(time))
    u_B[:STIM_beg] = u_B[STIM_end:] = 0

    u_ctxA = np.ones_like(time)*ctxA
    u_ctxA[:CTX_beg] = u_ctxA[CTX_end:] = 0

    u_ctxB = np.ones_like(time)*(1-ctxA)
    u_ctxB[:CTX_beg] = u_ctxB[CTX_end:] = 0

    go = 0
    if ctxA  and ma < 0:
      go = 1
    elif ctxA == 0 and mb > 0:
      go = -1

    return [ma,mb,ctxA,go,go],np.array([u_A,u_B,u_ctxA,u_ctxB])

### simulation helpers

def generate_stimuli_fixed(time, noise=0.1):
  trials = []
  for ctxA in [0,1]:
    for ma in [-1,1]:
      for mb in [-1,1]:
        trials.append(generate_stimuli(time, noise, ma=ma, mb=mb, ctxA=ctxA))

  return trials


## decoder helpers
def train_LD(X, y):
  ''' train linear discriminant on X (neurons x trials) data, based on y labels (1/0)'''

  y = y - np.mean(y)
  ct = np.mean(X.T[y>0,:],0)
  cr = np.mean(X.T[y<0,:],0)

  #w = np.linalg.pinv(X @ X.T) @ (ct - cr)  
  w = (ct - cr)  
  b = (ct*w + cr*w) / 2

  return w,b

def proj_LD(X,w,b):
  ''' project data X (neurons x trials) on hyper plane w,b'''
 # return np.mean(X *  w[:,None,None],0)
  return np.mean(X *  w[:,None,None]-  b[:,None,None],0)


def plot_all_decs(data,trial_stims,y,ctxs, n_pops = 3):
  ws = []
  for ci,ctx_i in enumerate(ctxs):
    for pi,pop in enumerate(data):
      i = ci*len(data) + pi + 1
      ax = plt.subplot(len(ctxs),len(data),i)

      w,b = train_LD(pop[:,ctx_i,-1],y[ctx_i])
      P=proj(pop[:,ctx_i],w,b)

      plot_sims(time,P,trial_stims[ctx_i],ax=ax)

      if ax.get_ylim()[1] < 1: ax.set_ylim(-1,1)
      if pi == 0: ax.set_ylabel("Context %i" % ci) 
      else: ax.set_yticks([])

      if ci == 0: 
        ax.set_title("#%i pop" % (pi+1))
        if pi+1 > n_pops: ax.set_title("all populations")

      ws.append(w)

  return ws


  ## plotting helpers

def plot_sigmas(Sigmas_):

  fig = plt.figure(figsize=(8,8))
  _axs = fig.subplots(nrows=2, ncols=2)
  fig.subplots_adjust(hspace=0.3)
  axs = _axs.flatten()

  Sigmas = np.copy(Sigmas_)

  for i in range(Sigmas[0].shape[0]):
    for j in range(i+1,Sigmas[0].shape[0]):
      Sigmas[:,j,i] = np.nan

  for s,sig in enumerate(Sigmas):
    im = axs[s].imshow(sig)
    fig.colorbar(im,ax=axs[s])
    axs[s].set_xticks(range(len(sig)))
    axs[s].set_yticks(range(len(sig)))
    axs[s].set_xticklabels(["n1", "m1", "w", "Ia","Ib", "ctxA", "ctxB"])
    axs[s].set_yticklabels(["n1", "m1", "w", "Ia","Ib", "ctxA", "ctxB"])

def performance(trial_stims, z,axs=None,fig=None):

  stims = np.array([a for a in np.array(np.array(trial_stims)[:,0])])
  ctx_idx = stims[:,2] == 1


  mmax = abs(z).max()
  mmin = -abs(z).max()
  biny=binx=[mmin,0,mmax]

  norm = cm.colors.Normalize(vmax=mmax, vmin=mmin)
  cmap = cm.coolwarm

  if not axs:
    fig = plt.figure(figsize=(8,4))
    _axs = fig.subplots(nrows=2, ncols=1)
    fig.subplots_adjust(hspace=0.3)
    axs = _axs.flatten()

  axs[0].set_title(r"$Ctx_B$")
  x=sts.binned_statistic_2d(stims[~ctx_idx,0],stims[~ctx_idx,1],z[~ctx_idx,-1],statistic='mean',bins=2)
  im=axs[0].imshow(x[0],cmap=cmap, norm=norm,origin='lower',extent =[mmin,mmax,mmin,mmax])
  cbar=fig.colorbar(im,ax=axs[0],shrink=.5,ticks=[mmin,0,mmax])
  cbar.ax.set_yticklabels(['Left', 'No Go', 'Right'])  # horizontal colorbar

  axs[0].plot([mmin,mmax],[0,0],"k--",alpha=0.2)
  axs[0].plot([0,0],[mmin,mmax],"k--",alpha=0.2)

#   axs[0].set_xticks([])
#   axs[0].set_yticks([])
 # axs[0].set_xlabel(r"$I_A$",fontsize=15)
  axs[0].set_ylabel(r"$I_A$ ",fontsize=15, rotation=0,labelpad=15)

  axs[1].set_title(r"$Ctx_A$")
  x=sts.binned_statistic_2d(stims[ctx_idx,0],stims[ctx_idx,1],z[ctx_idx,-1],statistic='mean',bins=2)
  im = axs[1].imshow(x[0],cmap=cmap, norm=norm,origin='lower',extent =[mmin,mmax,mmin,mmax])
  cbar=fig.colorbar(im,ax=axs[1],shrink=.5,ticks=[mmin,0,mmax])
  cbar.ax.set_yticklabels(['Left', 'No Go', 'Right'])  # horizontal colorbar

  axs[1].plot([mmin,mmax],[0,0],"k--",alpha=0.2)
  axs[1].plot([0,0],[mmin,mmax],"k--",alpha=0.2)

  axs[1].set_yticks([])
  axs[1].set_xticks([])
  axs[1].set_xlabel(r"$I_B$",fontsize=15)
  axs[1].set_ylabel(r"$I_A$",fontsize=15,rotation=0,labelpad=15)

  return axs
  
def performance2(trial_stims, z,axs=None,fig=None):

  stims = np.array([a for a in np.array(np.array(trial_stims)[:,0])])
  ctx_idx = stims[:,2] > 0.5

  stims = np.array([a for a in np.array(np.array(trial_stims)[:,-1])])

  mmax = abs(z).max()
  mmin = -abs(z).max()
  biny=binx=[mmin,0,mmax]

  norm = cm.colors.Normalize(vmax=mmax, vmin=mmin)
  cmap = cm.coolwarm

  if not axs:
    fig = plt.figure(figsize=(8,4))
    _axs = fig.subplots(nrows=2, ncols=1)
    fig.subplots_adjust(hspace=0.3)
    axs = _axs.flatten()

  axs[0].set_title(r"$Ctx_B$")
  x=sts.binned_statistic_2d(stims[~ctx_idx,0],stims[~ctx_idx,1],z[~ctx_idx,-1],statistic='mean',bins=2)
  im=axs[0].imshow(x[0],cmap=cmap, norm=norm,origin='lower',extent =[mmin,mmax,mmin,mmax])
  cbar=fig.colorbar(im,ax=axs[0],shrink=.5,ticks=[mmin,0,mmax])
  cbar.ax.set_yticklabels(['Left', 'No Go', 'Right'])  # horizontal colorbar

  axs[0].plot([mmin,mmax],[0,0],"k--",alpha=0.2)
  axs[0].plot([0,0],[mmin,mmax],"k--",alpha=0.2)

#   axs[0].set_xticks([])
#   axs[0].set_yticks([])
 # axs[0].set_xlabel(r"$I_A$",fontsize=15)
  axs[0].set_ylabel(r"$I_A$ ",fontsize=15, rotation=0,labelpad=15)

  axs[1].set_title(r"$Ctx_A$")
  x=sts.binned_statistic_2d(stims[ctx_idx,0],stims[ctx_idx,1],z[ctx_idx,-1],statistic='mean',bins=2)
  im = axs[1].imshow(x[0],cmap=cmap, norm=norm,origin='lower',extent =[mmin,mmax,mmin,mmax])
  cbar=fig.colorbar(im,ax=axs[1],shrink=.5,ticks=[mmin,0,mmax])
  cbar.ax.set_yticklabels(['Left', 'No Go', 'Right'])  # horizontal colorbar

  axs[1].plot([mmin,mmax],[0,0],"k--",alpha=0.2)
  axs[1].plot([0,0],[mmin,mmax],"k--",alpha=0.2)

  axs[1].set_yticks([])
  axs[1].set_xticks([])
  axs[1].set_xlabel(r"$I_B$",fontsize=15)
  axs[1].set_ylabel(r"$I_A$",fontsize=15,rotation=0,labelpad=15)

  return axs 

def plot_dynamics_populations(Sigmas, inputs_on,IA=1,IB=1,ax=None):
  if not ax:
    fig = plt.figure(figsize=(3.5,6))
    ax = fig.add_subplot(1,1,1)

  Sigmas = np.array(Sigmas.copy())
  # go through all possible inputs
  for input in range(IAi,IctxBi + 1):
    # if current input is not set ON, shut it OFF 
    if input not in inputs_on:
      Sigmas[:,input, input] = 0
      Sigmas[:,input,Ni] = 0

  n_pops = len(Sigmas)
  k_beg=-5
  k_end = 5
  kappas = np.linspace(k_beg, k_end, 500)

  dkdt = np.zeros([n_pops,len(kappas)])

  for ik, ka in enumerate(kappas):
      for pop in range(n_pops):


        #delta = Sigmas[pop][Mi,Mi]*(ka**2) + np.sum(Sigmas[pop][IAi:,IAi:]**2)

        delta =  Sigmas[pop][Mi,Mi] * (kappas[0]**2) + \
            np.sum(np.diagonal(Sigmas[pop])[IAi:])
            
        dkdt[pop,ik] = -ka + Sigmas[pop][Mi,Ni]*_phi_prime(0, delta)*ka + \
                        IA*Sigmas[pop][IAi,Ni]*_phi_prime(0, delta) + \
                        IB*Sigmas[pop][IBi,Ni]*_phi_prime(0, delta) + \
                        Sigmas[pop][IctxAi,Ni]*_phi_prime(0, delta) + \
                        Sigmas[pop][IctxBi,Ni]*_phi_prime(0, delta)

  Fk = np.mean(dkdt,0)
  Fk[Fk>0] = 10
  Fk[Fk<0] = -10
  kappasm1 = kappas[:-1]
  stable_fp = kappasm1[np.diff(Fk)<-1]+0.5*(kappas[1]-kappas[0])

  [ax.plot(kappas,dkdt[pop], label="pop %i" % pop) for pop in range(n_pops)]
  ax.plot(kappas,np.zeros_like(kappas),"k--",lw=1,alpha=0.2)
  ax.legend(loc='upper right')
  ax.set_ylabel(r"$\dot \kappa$",rotation=0,labelpad=10)
  ax.set_xlim([k_beg,k_end])
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.set_xlabel(r"$\kappa$")

  return ax
  
def do_effective_conn(Sigmas, kappas, inputs):

  n_pops = len(Sigmas)

  sigma_mm, sigma_mn, sigma_nIa, sigma_nIb, sigma_nIctxA, sigma_nIctxB  = np.zeros((6,n_pops))

  for pop in range(n_pops):
    delta = (Sigmas[pop][Mi,Mi]) * (kappas[0]**2) + \
          np.sum(np.diagonal(Sigmas[pop])[IAi:] * inputs**2)
    
    sigma_mn[pop]=Sigmas[pop][Mi,Ni] * _phi_prime(0,delta)
    sigma_mm[pop]=Sigmas[pop][Mi,Mi] * _phi_prime(0,delta)
    sigma_nIa[pop]=Sigmas[pop][Ni,IAi] * _phi_prime(0,delta)
    sigma_nIb[pop]=Sigmas[pop][Ni,IBi] * _phi_prime(0,delta)
    sigma_nIctxA[pop]=Sigmas[pop][Ni,IctxAi] * _phi_prime(0,delta)
    sigma_nIctxB[pop]=Sigmas[pop][Ni,IctxBi] * _phi_prime(0,delta)

  return [sigma_mm, sigma_mn, sigma_nIa, sigma_nIb,sigma_nIctxA, sigma_nIctxB]
  
def plot_dynamics_fixed_points(Sigmas, inputs_on,IA=1,IB=1,ax=None,kend=50,color='gray',label=None):

  inputs = np.zeros(4) 
  if IAi in inputs_on:
    inputs[0] = IA
  if IBi in inputs_on:
    inputs[1] = IB
  if IctxAi in inputs_on:
    inputs[2] = 1
  if IctxBi in inputs_on:
    inputs[3] = 1

  if not ax:
    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(1,1,1)

  n_pops = len(Sigmas)
  kbeg = -1*kend 
  kappas = np.linspace(kbeg, kend, 500)

  dkdt = np.zeros([n_pops,len(kappas)])

  for ik, ka in enumerate(kappas):
      eff = np.array(do_effective_conn(Sigmas, [ka], inputs))
      for pop in range(n_pops):

        sigma_mm,sigma_mn, sigma_nIa, sigma_nIb, \
          sigma_nIctxA, sigma_nIctxB = eff[:,pop]

        dkdt[pop,ik] = -ka + sigma_mn*ka + \
                          inputs[0]*sigma_nIa + \
                          inputs[1]*sigma_nIb + \
                          inputs[2]*sigma_nIctxA + \
                          inputs[3]*sigma_nIctxB

  Fk = np.mean(dkdt,0)
  Fk[Fk>0] = 10
  Fk[Fk<0] = -10
  kappasm1 = kappas[:-1]
  stable_fp = kappasm1[np.diff(Fk)<-1]+0.5*(kappas[1]-kappas[0])

  [ax.plot(kappas,dkdt[pop], label="pop %i" % pop,alpha=0.5) for pop in range(n_pops)]
  ax.plot(kappas,np.mean(dkdt,0),color=color,label=label)
  ax.plot(kappas,np.zeros_like(kappas),"k--",lw=1,alpha=0.2)
  ax.plot([0,0],[kbeg,kend],"k--",lw=1,alpha=0.2)
  ax.set_xlabel(r"$\kappa$")
  ax.set_ylabel(r"$\dot \kappa$", rotation=0,labelpad=10)
  ax.set_xlim([kbeg,kend])
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.scatter(stable_fp,np.zeros_like(stable_fp), edgecolor='C3', color='w', s=40, lw=2, zorder=4)
  return ax,stable_fp


def plot_dynamics_1_sigma(Sigmas,axs,color,label=None,kend=50,CTX=IctxAi):

  _,fps = plot_dynamics_fixed_points(Sigmas,[IAi,IBi,CTX],IA=-1,IB=-1,ax=axs[0],color=color,label=label,kend=kend)
  axs[0].set_title(r"$I_A<0,I_B<0$")
  axs[0].set_xlim(-kend,kend)
  axs[0].set_ylim(-kend,kend)
  axs[0].set_xticks(fps)
  # axs[0].set_yticks([])
  axs[0].legend(frameon=False)

  _,fps = plot_dynamics_fixed_points(Sigmas,[IAi,IBi,CTX],IA=-1,IB=1,ax=axs[1],color=color,label=label,kend=kend)
  axs[1].set_title(r"$I_A<0,I_B>0$")
  axs[1].set_xlim(-kend,kend)
  axs[1].set_ylim(-kend,kend)
  axs[1].set_xticks(fps)
  axs[1].set_yticks([])

  _,fps = plot_dynamics_fixed_points(Sigmas,[CTX],IA=0,IB=0,ax=axs[2],color=color,label=label,kend=kend)
  axs[2].set_title(r"$I_A=0,I_B=0$")
  axs[2].set_xlim(-kend,kend)
  axs[2].set_ylim(-kend,kend)
  axs[2].set_xticks(fps)
  axs[2].set_yticks([])

  _,fps = plot_dynamics_fixed_points(Sigmas,[IAi,IBi,CTX],IA=1,IB=-1,ax=axs[3],color=color,label=label,kend=kend)
  axs[3].set_title(r"$I_A>0,I_B<0$")
  axs[3].set_xlim(-kend,kend)
  axs[3].set_ylim(-kend,kend)
  axs[3].set_xticks(fps)
  axs[3].set_yticks([])

  _,fps = plot_dynamics_fixed_points(Sigmas,[IAi,IBi,CTX],IA=1,IB=1,ax=axs[4],color=color,label=label,kend=kend)
  axs[4].set_title(r"$I_A>0,I_B>0$")
  axs[4].set_xlim(-kend,kend)
  axs[4].set_ylim(-kend,kend)
  axs[4].set_xticks(fps)
  axs[4].set_yticks([])

  _,fps = plot_dynamics_fixed_points(Sigmas,[],IA=0,IB=0,ax=axs[5],color=color,label=label,kend=kend)
  axs[5].set_title(r"$I_A=0,I_B=0, IctxAi=0$")
  axs[5].set_xlim(-kend,kend)
  axs[5].set_ylim(-kend,kend)
  axs[5].set_xticks(fps)
  axs[5].set_yticks([])

FS = 8

def get_pop(pop,N,n_pops=3):
  if pop >= n_pops:
    return np.ones(N) == 1

  idx = np.zeros(N)
  pop_size = N // n_pops
  idx[pop*pop_size:(pop+1)*pop_size] = 1
  return idx == 1


def plot_sims_vs_sims(z1, z2, trial_stims,ax):


  for tr,(stim_par, stims) in enumerate(trial_stims):
    ls = "-"
    lw = 2
    if stim_par[0] > 0 and stim_par[1] > 0:
      lw = 1
      ls = "--"
    if stim_par[0] < 0 and stim_par[1] < 0:
      lw=1
      ls= "--"

    color = "gray"
    if stim_par[2] == 1 and stim_par[0] < 0:
      color = "indianred"
    elif stim_par[2] == 0 and stim_par[1] < 0: 
      color="royalblue"

    ax.plot(z1[tr], z2[tr], ls,c=color,lw=lw)
    #ax.plot(z1[tr], z2[tr], "o",c=color,alpha=0.5)

    format(ax,z1,z2)

def plot_sims(time, z, trial_stims,ax):

  for tr,(stim_par, stims) in enumerate(trial_stims):
    lw = 2
    ls = "-"
    if stim_par[0] > 0 and stim_par[1] > 0:
      lw = 1
      ls = "--"
    if stim_par[0] < 0 and stim_par[1] < 0:
      lw = 1
      ls = "--"

    color = "gray"
    if stim_par[2] == 1 and stim_par[0] < 0:
      color = "indianred"
    elif stim_par[2] == 0 and stim_par[1] > 0: 
      color="royalblue"

    ax.plot(time, z[tr], ls,c=color,lw=lw,alpha=0.75)

    format(ax,z)


def plot_sims_stim(time, z, trial_stims,ax,stim=0):

  for tr,(stim_par, stims) in enumerate(trial_stims):

    ls = "-"
    if stim_par[stim] < 0:
      ls = "--"

    ax.plot(time, z[tr],ls=ls,c="black",alpha=0.75)

    format(ax,z)

def get_trial_color(trial):
    stim_par, stims = trial 
    lw = 2
    ls = "-"
    if stim_par[0] > 0 and stim_par[1] > 0:
      lw = 1
      ls = "--"
    if stim_par[0] < 0 and stim_par[1] < 0:
      lw = 1
      ls = "--"

    color = "gray"
    if stim_par[2] == 1 and stim_par[0] < 0:
      color = "indianred"
    elif stim_par[2] == 0 and stim_par[1] < 0: 
      color="royalblue"


    return color,ls

def plot_sims_2ax(z1, z,trial_stims,ax):

  for tr,(stim_par, stims) in enumerate(trial_stims):
    lw = 2
    ls = "-"
    if stim_par[0] > 0 and stim_par[1] > 0:
      lw = 1
      ls = "--"
    if stim_par[0] < 0 and stim_par[1] < 0:
      lw = 1
      ls = "--"

    color = "gray"
    if stim_par[2] == 1 and stim_par[0] < 0:
      color = "indianred"
    elif stim_par[2] == 0 and stim_par[1] < 0: 
      color="royalblue"
    ax.plot(z2[tr], z1[tr], ls,c=color,lw=lw,alpha=0.75)
      

def format(ax,z1,z2=[]):
  if len(z2) < 1 :
    max_y=np.max(abs(np.concatenate(z1)))*1.5
    ax.plot([time[STIM_beg],time[STIM_beg]],[-max_y,max_y],"k--",lw=1,alpha=0.5)
    ax.plot(time,np.zeros_like(time),"k--",lw=1)
    ax.set_xlim([time[0],time[-1]])
    #ax.set_xticks([])
  else:
    max_x=np.max(abs(np.concatenate(z1)))*1.5
    max_y=np.max(abs(np.concatenate(z2)))*1.5
    ax.set_xlim([-max_x,max_x])

  ax.set_ylim([-max_y,max_y])
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
  ax.tick_params(labelsize=FS,direction='in')
  ax.yaxis.get_offset_text().set_fontsize(FS)

def generate_network_sigmas(Sigmas, N):

    n_pop = len(Sigmas)
    As = [gram_factorization(S) for S in Sigmas]

    pops = []

    for ni in range(n_pop):
      X = np.random.randn(len(Sigmas[0]),N//n_pop)
      X = gram_schmidt(X)
      X=(X.T / np.std(X,1)).T # following Manuel's code
      pop = As[ni] @ X
      pops.append(pop)

    return np.concatenate(pops,1)


def generate_2area_network(N,Sigmas,no_pop_0=False):
  # A1
  n1, m1, w, IA, IB, IctxA, IctxB =torch.tensor(generate_network_sigmas(Sigmas[0], N//2))[:,:,None].float()

  if no_pop_0:
    m1[2*N//6:] = 0

  Ik = torch.zeros((N,1))
  Ix = (IctxA - IctxB)/2
  Ik = (IctxA + IctxB)/2

  I_AP =  torch.normal(0, 1, size=(N//2,1))

  n_AP = torch.clone(m1)
  n_AP[2*N//6:] = 0

  A1 = m1 @ n1.T 
  A1_PFC = I_AP @ n_AP.T 
  IA1 = torch.column_stack([IA,IB,torch.zeros_like(Ix),torch.zeros_like(Ix)])

  # PFC
  n1p, m1p,Ip_ctxA, Ip_ctxB =torch.tensor(generate_network_sigmas([Sigmas[1]], N//2))[:,:,None].float()
  # Ip_ctxA /= torch.linalg.norm(Ip_ctxA)
  # Ip_ctxB /= torch.linalg.norm(Ip_ctxB)
  #m1p /= torch.linalg.norm(m1p)

  n_PA = torch.clone(m1p)

  PFC = m1p @ n1p.T
  # PFC_A1 = IctxA  @ Ip_ctxA.T +  IctxB  @ Ip_ctxB.T 
  PFC_A1 = Ix @ n_PA.T

  IPFC = torch.column_stack([torch.zeros_like(IA),torch.zeros_like(IA),Ip_ctxA,Ip_ctxB])


  # build J and input matrix 
  J = torch.zeros([N,N])
  J[:N//2,:N//2] = A1/(N//2)
  J[N//2:,N//2:] = PFC/(N//2)
  J[N//2:,:N//2] = A1_PFC/(N//3) #/N
  J[:N//2,N//2:] = PFC_A1/(N//3)

  J = torch.tensor(J).float()

  I = torch.row_stack([IA1,IPFC])

  return [J,I,IA1,IPFC,IA,IB,n1,n1p,m1,m1p,n_AP,n_PA,Ik,I_AP,Ix]

def plot_normalized_ff_fb(A1_X,PFC_X,vecs):

  J,I,IA1,IPFC,IA,IB,n1,n1p,m1,m1p,n_AP,n_PA,Ik,I_AP,Ix = vecs

  def normalize(vecs):
      for vi,ve in enumerate(vecs):
          vecs[vi] = ve / np.linalg.norm(ve)
      return vecs

  a1_vecs = torch.concat([torch.ones_like(m1),m1,Ix,n_AP],1).T
  pfc_vecs = torch.concat([torch.ones_like(m1p),m1p,n_PA,I_AP],1).T

  proj_a1 = A1_X.T @ normalize(a1_vecs).T
  proj_pfc = PFC_X.T @ normalize(pfc_vecs).T

  titles = ['mean','recurrent', 'feedback', 'feedfwd']
  plt.figure(figsize=(10,5))

  ylims = []
  axs = []
  for i,proj in enumerate(proj_a1.T):

      ax=plt.subplot(2,4,i+1)
      plt.title(titles[i])
      plt.plot(proj.T,color='black',alpha=0.1)
      axs.append(ax)
      ylims.append(plt.ylim())
      if i <1: plt.ylabel('A1')
      

      ax = plt.subplot(2,4,4+i+1)
      plt.plot(proj_pfc[:,:,i],color='black',alpha=0.1)
      ylims.append(plt.ylim())
      axs.append(ax)
      if i<1: plt.ylabel('PFC')
    
  ylim = max(np.abs(np.concatenate(ylims)))*1.5
  [ax.set_ylim(-ylim,ylim) for ax in axs]
  axs[0].set_ylim(-5,5)
  axs[1].set_ylim(-5,5)

  return axs
