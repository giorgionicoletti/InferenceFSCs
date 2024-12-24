import torch
from torch import nn
import numpy as np
import numba as nb
import random

import utils

class InferenceDiscreteObs:
    def __init__(self, FSC):
        """
        """
        self.FSC = FSC

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        theta = torch.tensor(self.FSC.theta.astype(np.float32), device=self.device)
        self.theta = nn.Parameter(theta)

        psi = torch.tensor(self.FSC.psi.astype(np.float32), device=self.device)
        self.psi = nn.Parameter(psi)

        self.rho = nn.Softmax(dim=0)(self.psi)
        self.TMat = utils.torch_softmax_2dim(self.theta, dims=(2, 3))

        self.InternalMemSpace = torch.arange(self.FSC.M)
        self.InternalActSpace = torch.arange(self.FSC.A)
        self.InternalObsSpace = torch.arange(self.FSC.Y)

        self.trajectories_loaded = False
        self.optimizer_initialized = False
        self.trained = False

    def load_trajectories(self, trajectories):
        """
        Loads a set of trajectories to be used for training the FSC.

        Parameters:
        --- trajectories: list of dicts
            List of dictionaries containing the actions and observations for each trajectory.
        """
        self.ObsAct_trajectories = []

        self.pStart_ya_emp = np.zeros((self.FSC.Y, self.FSC.A))   

        for trajectory in trajectories:
            observations = self.__map_obs_to_internal_space(trajectory["observations"])
            actions = self.__map_act_to_internal_space(trajectory["actions"])

            self.ObsAct_trajectories.append([torch.tensor(observations), torch.tensor(actions)])
            
            y0 = observations[0]
            a0 = actions[0]

            self.pStart_ya_emp[y0, a0] += 1
        
        self.pStart_ya_emp /= np.sum(self.pStart_ya_emp)
            
        self.trajectories_loaded = True

    def __map_obs_to_internal_space(self, obs):
        """
        Helper method to map an observation sequence to the internal observation space.

        Parameters:
        --- obs: np.array
            Observation sequence to map.

        Returns:
        --- obs_internal: np.array
            Observation sequence in the internal observation space.
        """
        return np.array([self.get_obs_idx(o) for o in obs])
    
    def __map_act_to_internal_space(self, act):
        """
        Helper method to map an action sequence to the internal action space.

        Parameters:
        --- act: np.array
            Action sequence to map.

        Returns:
        --- act_internal: np.array
            Action sequence in the internal action space.
        """
        return np.array([self.get_act_idx(a) for a in act])
        
    def get_obs_idx(self, obs):
        """
        Helper method to get the index of an observation in the observation space.

        Parameters:
        --- obs: int
            Observation to get the index of.

        Returns:
        --- idx: int
            Index of the observation in the observation space.
        """

        return np.where(obs == self.FSC.ObsSpace)[0][0]
    
    def get_act_idx(self, act):
        """
        Helper method to get the index of an action in the action space.

        Parameters:
        --- act: int
            Action to get the index of.

        Returns:
        --- idx: int
            Index of the action in the action space.
        """

        return np.where(self.FSC.ActSpace == act)[0][0]
    
    def get_mem_idx(self, mem):
        """
        Helper method to get the index of a memory state in the memory space.

        Parameters:
        --- mem: int
            Memory state to get the index of.

        Returns:
        --- idx: int
            Index of the memory state in the memory space
        """

        return np.where(self.FSC.MemSpace == mem)[0][0]
    
    def evaluate_nloglikelihood(self, idx_traj, grad_required=False):
        """
        Wrapper method to evaluate the negative log-likelihood of a given trajectory.
        It distinguishes between the case of custom observation and action spaces and the case
        of default observation and action spaces.

        Parameters:
        --- idx_traj: int
            Index of the trajectory to evaluate.
        --- grad_required: bool (default = False)
            Flag indicating whether the gradient is required or not.

        Returns:
        --- nLL: float
            Negative log-likelihood of the trajectory.
        """
        observations, actions = self.ObsAct_trajectories[idx_traj]
        
        return self.loss(observations, actions, grad_required = grad_required)
            
    def loss(self, observations, actions, grad_required=True):
        """
        Method to compute the negative log-likelihood of a given trajectory with default observation and action spaces.
        The gradients of the loss are computed if the grad_required flag is set to True.

        Parameters:
        --- observations: torch.tensor
            Array of observations.
        --- actions: torch.tensor
            Array of actions.
        --- grad_required: bool (default = True)
            Flag indicating whether the gradient is required or not.

        Returns:
        --- nLL: float
            Negative log-likelihood of the trajectory.
        """
        nLL = torch.tensor(0.0, requires_grad = grad_required)

        for t in range(observations.size(0)):
            idx_a = actions[t]
            idx_obs = observations[t]

            transition_probs = self.TMat[idx_obs, :, :, idx_a].T

            if t == 0:
                m = torch.matmul(transition_probs, self.rho)
            else:
                m = torch.matmul(transition_probs, m)

            mv = torch.sum(m)
            nLL = nLL - torch.log(mv)
            m /= mv

        return nLL - torch.log(torch.sum(m))       

    def optimize(self, NEpochs, NBatch, lr, train_split=0.8, optimizer=None, gamma=0.9, verbose=False,
                 maxiter=1000, rho0=None, th=1e-6, c_gauge = 0):
        """
        Method to optimize the parameters of the FSC using the loaded trajectories. The optimization is performed
        using the Adam optimizer with a learning rate schedule. The trajectories are split into a training and a
        validation set, and the loss is computed for both sets at each epoch. The training loss is computed over
        random batches of trajectories.

        """
        assert self.trajectories_loaded, "No trajectories have been loaded. Load trajectories with the load_trajectories method."
        assert self.trained == False, "The model has already been trained. If you want to train it again, reinitialize it or set the flag self.trained to False."

        if optimizer is not None:
            self.optimizer = optimizer([self.theta], lr = lr)
        else:
            self.optimizer = torch.optim.Adam([self.theta], lr = lr)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = gamma)

        NTrain = int(train_split * len(self.ObsAct_trajectories))
        NVal = len(self.ObsAct_trajectories) - NTrain

        trjs_train = self.ObsAct_trajectories[:NTrain]
        trjs_val = self.ObsAct_trajectories[NTrain:]

        losses_train = []
        losses_val = []

        print(f"Training with {NTrain} trajectories and validating with {NVal} trajectories.")

        for epoch in range(NEpochs):
            running_loss = 0.0
            random.shuffle(trjs_train)

            for idx in range(0, NTrain, NBatch):
                self.optimizer.zero_grad()
                loss = torch.tensor(0.0, requires_grad = True)

                self.TMat = utils.torch_softmax_2dim(self.theta, dims = (2, 3))

                if rho0 is None:
                    rho0 = np.ones(self.FSC.M)/self.FSC.M

                rho, _ = InferenceDiscreteObs.optimize_rho(self.FSC.Y, self.FSC.M, self.FSC.A,
                                                           self.TMat.detach().cpu().numpy(), self.pStart_ya_emp,
                                                           rho0, maxiter, th = th)
                
                self.rho = torch.tensor(rho.astype(np.float32), device = self.device)
                self.psi = nn.Parameter(torch.log(self.rho) + c_gauge)

                count = 0

                for idx_traj in range(idx, idx + NBatch):
                    if idx_traj < NTrain:
                        loss_traj = self.loss(trjs_train[idx_traj][0], trjs_train[idx_traj][1])
                        loss = loss + loss_traj
                        count += 1

                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()/count

                if verbose: 
                    print(f"\t Epoch {epoch + 1} - Batch {idx//NBatch + 1} - Loss: {loss.item()/count} - Learning rate: {self.optimizer.param_groups[0]['lr']}")
                
            running_loss = running_loss/(NTrain//NBatch)
            losses_train.append(running_loss)

            running_loss_val = 0.0

            for idx_traj in range(NVal):
                loss_val = torch.tensor(0.0, requires_grad = False)

                loss_traj_val = self.loss(trjs_val[idx_traj][0], trjs_val[idx_traj][1], grad_required = False)
                loss_val = loss_val + loss_traj_val

                running_loss_val += loss_val.item()

            running_loss_val = running_loss_val/NVal
            losses_val.append(running_loss_val)

            print(f"Epoch {epoch + 1} - Training loss: {running_loss}, Validation loss: {running_loss_val} - Learning rate: {self.optimizer.param_groups[0]['lr']}")

            scheduler.step()
            
        self.trained = True

        return losses_train, losses_val

    @staticmethod
    @nb.njit
    def optimize_rho(Y, M, A, TMat, pya, rhok, maxiter, th):
        wVec = np.zeros((Y, A, M))
        for y in range(Y):
            for a in range(A):
                for m in range(M):
                    wVec[y, a, m] = np.sum(TMat[y, m, :, a])

        for _ in range(maxiter):
            wsumexp_test_k = np.zeros((Y, A))
            
            for y in range(Y):
                for a in range(A):
                    wsumexp_test_k[y, a] = np.sum(wVec[y, a] * rhok)
            
            grad = wVec * rhok / wsumexp_test_k[..., None]

            rhok_new = np.zeros(M)

            for y in range(Y):
                for a in range(A):
                    rhok_new += pya[y, a] * grad[y, a]
            
            # if np.linalg.norm(rhok_new - rhok) < th:
            #     break

            rhok = rhok_new

        return rhok, np.linalg.norm(rhok_new - rhok)

    def optimize_psionly(self, maxiter=1000, rho0=None, th=1e-6, c_gauge = 0):
        assert self.trajectories_loaded, "No trajectories have been loaded. Load trajectories with the load_trajectories method."
        assert self.trained == False, "The model has already been trained. If you want to train it again, reinitialize it or set the flag self.trained to False."

        if rho0 is None:
            rho0 = np.ones(self.FSC.M)/self.M

        rho, err = InferenceDiscreteObs.optimize_rho(self.FSC.Y, self.FSC.M, self.FSC.A,
                                                     self.TMat.detach().cpu().numpy(), self.pStart_ya_emp,
                                                     rho0, maxiter, th = th)

        self.rho = torch.tensor(rho.astype(np.float32), device = self.device)
        self.psi = nn.Parameter(torch.log(self.rho) + c_gauge)
        
        self.trained = True

        return err

    # def load_theta(self, theta):
    #     """
    #     Loads a new set of parameters for the transition probabilities of the FSC.

    #     Parameters:
    #     --- theta: torch.tensor of shape (Y, M, M, A)
    #         New parameters for the transition probabilities.
    #     """
    #     self.theta = nn.Parameter(torch.tensor(theta, device = self.device))
    #     self.TMat = fun.torch_softmax_2dim(self.theta, dims = (2, 3))
    #     self.policy = torch.sum(self.TMat, dim = 2)

    # def load_psi(self, psi):
    #     """
    #     Loads a new set of parameters for the initial memory occupation of the FSC.

    #     Parameters:
    #     --- psi: torch.tensor of shape (M)
    #         New parameters for the initial memory occupation.
    #     """
    #     self.psi = nn.Parameter(torch.tensor(psi, device = self.device))
    #     self.rho = nn.Softmax(dim = 0)(self.psi)