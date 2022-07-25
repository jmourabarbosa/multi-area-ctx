
import torch
import torch.nn as nn
from pathlib import Path
import os
import numpy as np
import json

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, hidden = self.lstm(x)
        x = self.linear(out)
        return x, out

class CTRNN(nn.Module):
    """Continuous-time RNN.

    Args:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons

    Inputs:
        input: (seq_len, batch, input_size), network input
        hidden: (batch, hidden_size), initial hidden activity
    """

    def __init__(self, input_size, hidden_size,tf,rank=1,dt=None, noise=1, device='cpu',**kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = 100
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        self.alpha = alpha
        self.oneminusalpha = 1 - alpha
        self.tf = tf

        #self.input2h = nn.Linear(input_size, hidden_size,bias=False)
        #self.input2h.requires_grad_(False)

        # self.input_stim = nn.Parameter(torch.randn(2, hidden_size))#.to(device)#.detach()
        # self.input_stim.requires_grad_(False)
        # self.input_ctx= nn.Parameter(torch.randn(2, hidden_size))#.to(device)#.detach()

        self.IA = nn.Parameter(torch.randn(1,hidden_size).type(torch.float))
        self.IA.requires_grad_(False)

        self.IB =nn.Parameter(torch.randn(1,hidden_size).type(torch.float))
        self.IB.requires_grad_(False)

        self.IXA = nn.Parameter(torch.randn(1,hidden_size).type(torch.float))
        self.IXB =nn.Parameter(torch.randn(1,hidden_size).type(torch.float))
        
        #self.IXA.requires_grad_(False)
        #self.IXB.requires_grad_(False)


        #self.m = nn.Parameter(torch.randn(hidden_size,rank).type(torch.float)/self.hidden_size)
        #self.n =nn.Parameter(torch.randn(hidden_size,rank).type(torch.float)/self.hidden_size)
        
        self.m = nn.Parameter(torch.randn(hidden_size).type(torch.float)/self.hidden_size)
        self.n =nn.Parameter(torch.randn(hidden_size).type(torch.float)/self.hidden_size)
        
        
        #self.m2 = nn.Parameter(torch.randn(hidden_size).type(torch.float)/self.hidden_size)
        #self.n2 =nn.Parameter(torch.randn(hidden_size).type(torch.float)/self.hidden_size)


        #self.m.requires_grad_(False)
        #self.n.requires_grad_(False)


        self.noise = noise
        self.device = device
        
    def init_hidden(self, input_shape):
        batch_size = input_shape[1]

        return torch.zeros(batch_size, self.hidden_size) 

    def recurrence(self, input, hidden):
        

        stims_input,ctx_input = input[:,:2],input[:,2:]
        A,B,XA,XB = input[:,0],input[:,1],input[:,2],input[:,3]
       # W = torch.mm(self.m,self.n.T) #+ 
        W = torch.outer(self.m,self.n.T)

        #pre_activation = torch.mm(stims_input,self.input_stim) + torch.mm(ctx_input,self.input_ctx) + torch.mm(torch.outer(self.m,self.n.T),torch.tanh(hidden).T).T 

        
        pre_activation =A[:,None] @ self.IA + B[:,None] @ self.IB + \
                        XA[:,None] @ self.IXA + XB[:,None] @ self.IXB + \
                        torch.mm(W,self.tf(hidden).T).T 

        rec_noise = torch.randn(self.hidden_size)*self.noise
        rec_noise = torch.randn(pre_activation.shape)*self.noise
        pre_activation += rec_noise.to(input.device)
        
       # pre_activation = self.input2h(input) + torch.mm(torch.matmul(self.m[:,None],self.n[None,:]),torch.tanh(hidden).T).T 

        #h_new = torch.tanh(hidden) * self.oneminusalpha + pre_activation * self.alpha
        h_new = hidden * self.oneminusalpha + pre_activation * self.alpha
        return h_new,pre_activation

    def forward(self, input, hidden=None):
        """Propogate input through the network."""
        if hidden is None:
            hidden = self.init_hidden(input.shape).to(input.device)

        output = []
        steps = range(input.size(0))
        X = []
        for i in steps:
            hidden, x = self.recurrence(input[i], hidden)
            output.append(hidden)
            X.append(x)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())
        X = torch.cat(X, 0).view(input.size(0), *X[0].size())

        return output, hidden, X


class RNNNet(nn.Module):
    """Recurrent network model.

    Args:
        input_size: int, input size
        hidden_size: int, hidden size
        output_size: int, output size
        rnn: str, type of RNN, lstm, rnn, ctrnn, or eirnn
    """
    def __init__(self, input_size, hidden_size, output_size, noise, tf=torch.tanh,device="cpu",**kwargs):
        super().__init__()

        # Continuous time RNN
        self.rnn = CTRNN(input_size, hidden_size, tf,noise=noise,device=device,**kwargs)
        self.fc =  nn.Parameter(torch.randn(hidden_size, output_size)/hidden_size) #nn.Linear(hidden_size, output_size)
        #self.fc = nn.Linear(hidden_size, output_size)
        self.tf = tf
        self.fc.requires_grad_(False)

    def forward(self, x):
        rnn_activity, _, X = self.rnn(x)
        out = rnn_activity @ self.fc

        #out = self.fc(rnn_activity)

        return out, rnn_activity
        
    def set_vecs(self,n1, m1, w, IA, IB, IctxA, IctxB):
    
        self.fc = nn.Parameter(w)
        self.rnn.n = nn.Parameter(n1)
        self.rnn.m = nn.Parameter(m1)
        self.rnn.IA = nn.Parameter(IA)
        self.rnn.IB = nn.Parameter(IB)
        self.rnn.IXA = nn.Parameter(IctxA)
        self.rnn.IXB = nn.Parameter(IctxB)



        
def test_net(net=None,env=None,envid=None,num_trials=100,device="cuda"):

  with torch.no_grad():

    # if no network provided, load it 
    # and infer environment from envid
    if net == None:
      assert envid, env
      modelpath = get_modelpath(envid)

      with open(modelpath / 'config.json') as f:
          config = json.load(f)
          
      net = RNNNet(input_size=env.observation_space.shape[0],
              hidden_size=config['hidden_size'],
              output_size=env.action_space.n,noise=config['noise'])
      net.load_state_dict(torch.load(modelpath / 'net.pth'))

    all_obs = []
    all_gts = []
    activity = []
    all_choices=[]
    X = []
    for i in range(num_trials):
      env.new_trial()
      ob, gt = env.ob, env.gt
      inputs = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float)
      action_pred, hidden = net(inputs.to(device))

      # Compute performance
      action_pred = action_pred.cpu().detach().numpy()
      #choice = action_pred[-1, 0, 0]
      choice = action_pred[-1, 0]
      all_choices.append(choice)
      all_gts.append(gt[-1])
      correct = abs(choice-gt[-1])
      all_obs.append(ob)

      # Log stimulus period activity
      activity.append(hidden.cpu().detach().numpy()[:, 0, :])
      #X.append(x.cpu().detach().numpy()[:, 0, :])

    act = torch.tensor(activity).cpu().type(torch.float)
    #inputs = torch.tensor(X).cpu().type(torch.float)
    z =  act @ net.rnn.m.detach().cpu().type(torch.float)
    gts = np.array(all_gts)
    choices = np.array(all_choices)
    obs = np.array(all_obs)

  return z, act, obs, gts, choices
 
def get_modelpath(envid):
    # Make a local file directories
    path = Path('nets/') / 'files'
    os.makedirs(path, exist_ok=True)
    path = path / envid
    os.makedirs(path, exist_ok=True)
    return path

def make_vecs(net):
    """
    return a list of vectors (list of numpy arrays of shape n) composing a network
    """
    inputs = np.squeeze([v.detach().cpu().numpy() for v in [net.rnn.IA, net.rnn.IB,net.rnn.IXA,net.rnn.IXB]])
    w = list(net.fc.detach().cpu().numpy().T)
    
    
    return np.array([net.rnn.n.detach().cpu().numpy(), net.rnn.m.detach().cpu().numpy()] + w + list(inputs))
    return np.array(list(net.rnn.n.detach().cpu().numpy().T) +  list(net.rnn.m.detach().cpu().numpy().T) + w + list(inputs)).T


def load_net(envid, env,device="cpu"):
    
    modelpath = get_modelpath(envid)
    with open(modelpath / 'config.json') as f:
      config = json.load(f)
    net = RNNNet(input_size=env.observation_space.shape[0],
          hidden_size=config['hidden_size'],
          output_size=env.action_space.n,noise=config['noise'],dt=config['dt'])
    net.load_state_dict(torch.load(modelpath / 'net.pth',  map_location=torch.device(device)))

    return net
    
