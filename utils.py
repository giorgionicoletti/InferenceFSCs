import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import pickle
import numpy as np
import random
from scipy.special import softmax as softmax
from IPython.display import display, clear_output
from itertools import islice


sm1 = nn.Softmax(dim = 1)

def wrapper_map_f_bin(bins_f, min_f, max_f):
    df = (max_f - min_f) / bins_f
    def map_f_bin(f):
        idx_f = np.clip( (f-min_f) / df, 0, bins_f-1)
        return int(idx_f)
    return map_f_bin

def Tmm(theta, f, a):
    F,M,M,A = theta.shape
    print(theta.dtype(),f.dtype())
    W = torch.einsum('fmna, f -> mna', theta, f)
    T = sm1(W.view(M,-1)).view(M,M,A)
    return T[:,:,a]

def M_SteadyState_f(theta, f, epsilon=0.000001):
    F,M,M,A = theta.shape
    with torch.no_grad():
        W = torch.einsum('fmna, f -> mna', theta, f)
        T = sm1(W.view(M,-1)).view(M,M,A).sum(2).T
        V, U = np.linalg.eig(T)
        mask = np.abs(np.abs(V)-1.0) < epsilon
        m_av = U[:,mask].reshape(-1,1)
        return np.abs(m_av) / np.abs(m_av).sum()
    
#Loss computed in one trajectory
def one_traj_loss(theta, f_traj, a_traj, m, T):
    m_update = 0.
    loss_trj = torch.tensor(0.0, requires_grad = True)
    for f, a in zip(f_traj[:T], a_traj[:T-1]):
        m_update += m.detach().numpy()
        m = torch.matmul(Tmm(theta, f, a).T, m)
        mv = torch.sum(m)
        loss_trj = loss_trj - torch.log(mv)
        m = m / mv
    loss_trj = loss_trj - torch.log(m.sum())
    return loss_trj, m_update

#Loss evaluation with dual parameter set!
#Loss evaluated over all the trajectories!

def trajs_loss_eval(theta, psi, batch, trajectories,grad_required=True):
    # shuffle gradient position
    x_pos_shuffle = False
    x_max_shuffle = 5
    maxT = 5000
    sm0 = nn.Softmax(dim = 0)
    loss = torch.tensor(0.0, requires_grad=grad_required)
    F, M, M, A = theta.shape

    with torch.set_grad_enabled(grad_required):
        for j, trj in enumerate(trajectories):
            
            # Load each trajectory
            # ----------------------------------------
            y_traj = trj["observations"]
            # ----------------------------------------
            a_traj = trj["actions"]  #.astype(int)


            x_shift = x_pos_shuffle * (np.random.rand() - 0.5) * x_max_shuffle

            #i dont know what is this
            f_traj = torch.tensor(np.column_stack((np.ones(y_traj.shape[0]),y_traj + x_shift)),requires_grad=True)
            T = min(f_traj.shape[0], maxT)

            # Initialize memory as tensor, memory is dependent on parameters psi
            
            rho = sm0(psi).double() #initial memory occupation

            loss_trj, _ = one_traj_loss(theta, f_traj, a_traj, rho, T)    
            loss = loss + loss_trj
    return loss/len(trajectories)

def Tmm(theta, f, a):
    F,M,M,A = theta.shape
    W = torch.einsum('fmna, f -> mna', theta.double(), f)
    T = sm1(W.view(M,-1)).view(M,M,A)
    return T[:,:,a]

#Theta parameters of the FSC:
def get_trajectories(source,traj_length,traj_size):
    #produces a dictionary of trajectories containing actions, observations, memory
    
    if source=='theta_param':
        trajectories=get_traj_from_theta(F,M,A,signal)
        
    if source=='q1q2_model':
        trajectories=get_traj_from_q1q2_model(w1,b1,w2,b2,alpha,signal)
        
    if source=='data':
        trajectorie=get_traj_from_data(file_name)
        
    return trajectories

#signal landscapes
def get_signal_landscape(kind,traj_length,traj_size):
    if kind=='step_like':
        with open("./obs_landscapes_step.pkl", "rb") as f:
            signal_dict = pickle.load(f)
    return signal_dict

def get_traj_from_theta(F,M,A,signal,random_theta,Ntraj,traj_len):
    if random_theta:
        theta = np.random.rand(F,M,M,A)
        theta = 2*theta-1
    else:
        if M==2:
            print('M=2')
            theta=np.array([[[[-0.61114431,-0.55987562],[ 0.59255672,0.82834098]],[[-0.88220033,0.46330714],[ 0.43192955,-0.52351326]]],
               [[[ 0.41101729,-0.31575377],[-0.64954147,0.36582982]],[[-0.24503104,0.00449758],[ 0.8396274,0.59647384]]]])
        elif M==1:
            print('M=1')
            theta=np.array([[[[-0.8556, -0.8582]]],[[[ 0.6520,  0.4511]]]])
    dict_trajs=[]
    for i in range(Ntraj):
        all_a=[]
        if M==1:
            m=0
        else:
            m=np.random.choice(M,p=[0.5,0.5])
        all_m=[m]
        y_traj=signal[i]['observations']
        f_traj = np.column_stack((np.ones(y_traj.shape[0]),y_traj))
        for kstep in range(traj_len):
            pi = pi_f(theta,f_traj[kstep],m)
            am = np.random.choice(M*A, p=pi)
            m = am // A
            a = am % A
            all_m.append(m)
            all_a.append(a)
        trajs={'observations': np.array(y_traj), 'actions': np.array(all_a), 'memories': np.array(all_m) }
        dict_trajs.append(trajs)
        print(i+1, "/", Ntraj)
        clear_output(wait=True)
    with open(f"./trajectories_sample_len{traj_len}_N{Ntraj}_fromtheta.pkl", "wb") as f:
        frames_num = pickle.dump(dict_trajs, f)
    return dict_trajs,theta

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
        

def pi_f(th, f, m):
    F, M, M, A = th.shape
    W = np.einsum('fmna, f -> mna', th, f)
    pi_fi = softmax(W.reshape(M,-1)[m]).reshape(-1)
    return pi_fi

def np_Tmm(theta, f, a):
    F,M,M,A = theta.shape
    W = np.einsum('fmna, f -> mna', theta, f)
    T = softmax(W.reshape(M,-1)).reshape(M,M,A)
    return T[:,:,a]

def np_M_SteadyState_f(theta, f, epsilon=0.000001):
    F,M,M,A = theta.shape
    W = np.einsum('fmna, f -> mna', theta, f)
    T = softmax(W.reshape(M,-1), axis=1).reshape(M,M,A).sum(2).T
    V, U = np.linalg.eig(T)
    mask = np.abs(np.abs(V)-1.0) < epsilon
    m_av = U[:,mask].reshape(-1,1)
    return np.abs(m_av) / np.abs(m_av).sum()

def pi_f_tot(th, f):
    F, M, M, A = th.shape
    W = np.einsum('fmna, f -> mna', th, f)
    pi_f = softmax(W.reshape(M,-1),axis=1).reshape(M,M,A)
    return pi_f


    
def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while (batch := tuple(islice(it, n))):
        yield batch

        