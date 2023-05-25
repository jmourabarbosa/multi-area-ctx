
from scipy import linalg
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

## some helpers

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

def aprox_PSD(G):
 
    w, v = np.linalg.eigh(G)
    if any(w<0):
      print('not PSD, truncated negative evals to 0, %f' % min(w))
      w[w<0] = 0
    x = v * np.sqrt(w)
    return x

def generate_low_rank_network(Sigmas, N):
    """ Sigmas: a list of vector covariances
        N: number of neurons
    """

    n_pop = len(Sigmas)
    As = [aprox_PSD(S) for S in Sigmas]

    pops = []

    for ni in range(n_pop):
      X = np.random.randn(len(Sigmas[0]),N//n_pop)
      X = gram_schmidt(X)
      X=(X.T / np.std(X,1)).T 
      pop = As[ni] @ X
      pops.append(pop)

    return np.concatenate(pops,1)

# parameters
N=1000
g= 2
n_trials = 10
n_time = 2000
tau = 1
dt = torch.tensor(0.01)
device = "cpu"

# low rank network parameters
PFC_Sigma = np.eye(2,2)
Ni,Mi = 0,1
PFC_Sigma[Ni,Mi] = PFC_Sigma[Mi,Ni] =2
PFC_Sigma[Ni,Ni] = PFC_Sigma[Mi,Mi] =2

# generate low rank network vectors and J matrix
vecs =generate_low_rank_network([PFC_Sigma], N)
m,n = vecs[:2]

J = m[:,None] @ n[:,None].T 
J=torch.tensor(J).to(device)

# random component to get chaotic dynamics
Xi =g* torch.randn(N,N)/np.sqrt(N)

# data structures to save each time step for chaotic (X_r) and non-chaotic simulations
X = torch.zeros(N,n_trials,n_time)
X_r = torch.zeros(N,n_trials,n_time)

for tr in tqdm(range(n_trials)):

    # initialize networks close to the attractors (-1,1)
    c = np.random.choice([-1,1])  + torch.randn(1)*2
    x = torch.zeros(N).to(device).double() 
    x[:] = c*m
    x_r = x

    for ti in range(n_time):

        # simulate without chaotic component
        x = x + dt/tau*(-x + J/N  @ torch.tanh(x)) #+noise_eta[:,ti])
        X[:,tr,ti] = x

        # simulate w/ chaotic component, Xi
        x_r = x_r + dt/tau*(-x_r + (J/N + Xi) @ torch.tanh(x_r)) #+noise_eta[:,ti])
        X_r[:,tr,ti] = x_r


## plot simulations

plt.figure(figsize=(5,3))

plt.subplot(2,2,1)
plt.plot(X[:10,0,:].T)
plt.ylabel("activations")
plt.title("g=0")

plt.subplot(2,2,2)
kappa = X.T @ m / (np.linalg.norm(m)**2)
plt.plot(kappa,"red")
plt.ylabel("kappa")
plt.tight_layout()


plt.figure(figsize=(5,3))

plt.subplot(2,2,3)
plt.plot(X_r[:10,0,:].T)
plt.ylabel("activations")
plt.title("g=%f" % g)


plt.subplot(2,2,4)
kappa = X_r.T @ m / (np.linalg.norm(m)**2)
plt.plot(kappa,"red")
plt.ylabel("kappa")

plt.tight_layout()