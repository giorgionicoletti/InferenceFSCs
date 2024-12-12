import numpy as np
import numba as nb
import fun
import torch
from torch import nn
import random

import matplotlib.pyplot as plt


class InferenceDiscreteObs():

    def __init__(self, M, A, Y,
                 ObsSpace = None, ActSpace = None, MemSpace = None,
                 seed = None, minus1_opt = False):
        """
        Creates an instance of a Finite State Controller (FSC) for discrete observations. The parameters
        M, A, and Y are used to define the number of memory states, actions, and observations, respectively.
        The FSC is then trained to learn the transition probabilities and the initial memory occupation from
        a set of trajectories, by minimizing the negative log-likelihood of the observed trajectories.

        The optimization is performed using PyTorch, and the model is trained using the Adam optimizer with
        a learning rate schedule.

        Parameters:
        --- M: int
            Number of memory states.
        --- A: int
            Number of actions.
        --- Y: int
            Number of observations.
        --- ObsSpace: list (default = None)
            List of observations. If None, the observations are assumed to be integers from 0 to Y-1.
        --- ActSpace: list (default = None)
            List of actions. If None, the actions are assumed to be integers from 0 to A-1.
        --- MemSpace: list (default = None)
            List of memory states. If None, the memory states are assumed to be integers from 0 to M-1.
        --- seed: int (default = None)
            Seed for the random number generator. If None, the seed is not set.
        --- minus1_opt: bool (default = False)
            If True, the model learns the parameters of the FSC with M-1 memory states, and the initial memory occupation
            is set to 1 for the first memory state and learned for the others. This is useful to avoid identifiability
            issues in the optimization.
        """

        self.M = M
        self.A = A
        self.Y = Y

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
    
        if seed is not None:
            torch.manual_seed(seed)

        self.theta = nn.Parameter(torch.randn(self.Y, self.M, self.M, self.A, device = self.device))

        if minus1_opt:
            self.psi_toopt = nn.Parameter(torch.randn(self.M - 1, device = self.device))

            self.psi = torch.concatenate([torch.ones(1, device = self.device), self.psi_toopt],
                                          dim = 0)
        else:
            self.psi = nn.Parameter(torch.randn(self.M, device = self.device))

        self.minus1_opt = minus1_opt

        if ObsSpace is not None:
            assert len(ObsSpace) == self.Y, "The number of observations in ObsSpace must be equal to Y."
            self.ObsSpace = torch.tensor(ObsSpace)
            self.custom_obs_space = True
        else:
            self.ObsSpace = torch.arange(self.Y)
            self.custom_obs_space = False

        if ActSpace is not None:
            assert len(ActSpace) == self.A, "The number of actions in ActSpace must be equal to A."
            self.ActSpace = torch.tensor(ActSpace)
            self.custom_act_space = True
        else:
            self.ActSpace = torch.arange(self.A)
            self.custom_act_space = False

        if MemSpace is not None:
            assert len(MemSpace) == self.M, "The number of memory states in MemSpace must be equal to M."
            self.MemSpace = torch.tensor(MemSpace)
            self.custom_mem_space = True
        else:
            self.MemSpace = torch.arange(self.M)
            self.custom_mem_space = False

        self.rho = nn.Softmax(dim = 0)(self.psi)
        self.TMat = fun.torch_softmax_2dim(self.theta, dims = (2, 3))
        self.policy = torch.sum(self.TMat, dim = 2)

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

        for trajectory in trajectories:
            self.ObsAct_trajectories.append([torch.tensor(trajectory["observations"]),
                                             torch.tensor(trajectory["actions"])])
            
        self.trajectories_loaded = True

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
        if self.custom_obs_space:
            assert obs in self.ObsSpace, f"Observation {obs} is not in the observation space."
            return torch.where(self.ObsSpace == obs)[0][0]
        else:
            return obs
    
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
        if self.custom_act_space:
            assert act in self.ActSpace, f"Action {act} is not in the action space."
            return torch.where(self.ActSpace == act)[0][0]
        else:
            return act

    def evaluate_nloglikelihood(self, idx_traj, grad_required = False):
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
        if self.custom_obs_space or self.custom_act_space:
            return self.loss_customspace(observations, actions, grad_required = grad_required)
        else:
            return self.loss(observations, actions, grad_required = grad_required)
        
    def loss_customspace(self, observations, actions, grad_required = True):
        """
        Method to compute the negative log-likelihood of a given trajectory with custom observation and action spaces.
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
            idx_a = torch.where(self.ActSpace == actions[t])[0][0]
            idx_obs = torch.where(self.ObsSpace == observations[t])[0][0]

            transition_probs = self.TMat[idx_obs, :, :, idx_a].T

            if t == 0:
                m = torch.matmul(transition_probs, self.rho)
            else:
                m = torch.matmul(transition_probs, m)

            mv = torch.sum(m)
            nLL = nLL - torch.log(mv)
            m /= mv

        return nLL - torch.log(torch.sum(m))
    
    def loss(self, observations, actions, grad_required = True):
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


    def optimize(self, NEpochs, NBatch, lr, train_split = 0.8, optimizer = None, gamma = 0.9):
        """
        Method to optimize the parameters of the FSC using the loaded trajectories. The optimization is performed
        using the Adam optimizer with a learning rate schedule. The trajectories are split into a training and a
        validation set, and the loss is computed for both sets at each epoch. The training loss is computed over
        random batches of trajectories.

        Parameters:
        --- NEpochs: int
            Number of epochs for the optimization.
        --- NBatch: int
            Number of trajectories per batch.
        --- lr: float
            Initial learning rate for the optimizer.
        --- train_split: float (default = 0.8)
            Fraction of trajectories to use for training.
        --- optimizer: torch.optim (default = None)
            Optimizer to use for the optimization. If None, the Adam optimizer is used.
        --- gamma: float (default = 0.9)
            Decay factor for the learning rate schedule.

        Returns:
        --- losses_train: list
            List of training losses at each epoch.
        --- losses_val: list
            List of validation losses at each epoch.
        """
        assert self.trajectories_loaded, "No trajectories have been loaded. Load trajectories with the load_trajectories method."
        assert self.trained == False, "The model has already been trained. If you want to train it again, reinitialize it or set the flaf self.trained to False."

        if optimizer is not None:
            self.optimizer = optimizer([self.theta, self.psi], lr = lr)
        else:
            self.optimizer = torch.optim.Adam([self.theta, self.psi], lr = lr)

        # add an exponential scheduler
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

                self.TMat = fun.torch_softmax_2dim(self.theta, dims = (2, 3))
                self.rho = nn.Softmax(dim = 0)(self.psi)

                count = 0

                for idx_traj in range(idx, idx + NBatch):
                    if idx_traj < NTrain:
                        # here we go without the custom space for simplicity
                        loss_traj = self.loss(trjs_train[idx_traj][0], trjs_train[idx_traj][1])
                        loss = loss + loss_traj
                        count += 1

                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()/count

                # normalize the parameters using nn.functional.normalize
                self.psi.data = nn.functional.normalize(self.psi.data, p = 2, dim = 0)
                self.theta.data = nn.functional.normalize(self.theta.data, p = 2, dim = 0)

                print(f"\t Epoch {epoch + 1} - Batch {idx//NBatch + 1} - Loss: {loss.item()/count} - Learning rate: {self.optimizer.param_groups[0]['lr']}")
                
            running_loss = running_loss/(NTrain//NBatch)
            losses_train.append(running_loss)

            scheduler.step()

            running_loss_val = 0.0

            for idx_traj in range(NVal):
                loss_val = torch.tensor(0.0, requires_grad = False)

                loss_traj_val = self.loss(trjs_val[idx_traj][0], trjs_val[idx_traj][1], grad_required = False)
                loss_val = loss_val + loss_traj_val

                running_loss_val += loss_val.item()

            running_loss_val = running_loss_val/NVal
            losses_val.append(running_loss_val)

            print(f"Epoch {epoch + 1} - Training loss: {running_loss}, Validation loss: {running_loss_val}")

        self.trained = True

        return losses_train, losses_val

    def optimize_psionly(self, NEpochs, NBatch, lr, train_split = 0.8, optimizer = None, gamma = 0.9,
                         verbose = False):
        """
        Method to optimize the initial memory occupation of the FSC using the loaded trajectories, while keeping
        the transition probabilities fixed. The optimization is performed using the Adam optimizer with a learning
        rate schedule. The trajectories are split into a training and a validation set, and the loss is computed for
        both sets at each epoch. The training loss is computed over random batches of trajectories.

        Parameters:
        --- NEpochs: int
            Number of epochs for the optimization.
        --- NBatch: int
            Number of trajectories per batch.
        --- lr: float
            Initial learning rate for the optimizer.
        --- train_split: float (default = 0.8)
            Fraction of trajectories to use for training.
        --- optimizer: torch.optim (default = None)
            Optimizer to use for the optimization. If None, the Adam optimizer is used.
        --- gamma: float (default = 0.9)
            Decay factor for the learning rate schedule.

        Returns:
        --- losses_train: list
            List of training losses at each epoch.
        --- losses_val: list
            List of validation losses at each epoch.
        """
        assert self.trajectories_loaded, "No trajectories have been loaded. Load trajectories with the load_trajectories method."
        assert self.trained == False, "The model has already been trained. If you want to train it again, reinitialize it or set the flaf self.trained to False."

        if optimizer is not None:
            self.optimizer = optimizer([self.psi], lr = lr)
        else:
            if self.minus1_opt:
                self.optimizer = torch.optim.Adam([self.psi_toopt], lr = lr)
            else:
                self.optimizer = torch.optim.Adam([self.psi], lr = lr)

        # add an exponential scheduler
        scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = gamma)

        NTrain = int(train_split * len(self.ObsAct_trajectories))
        NVal = len(self.ObsAct_trajectories) - NTrain

        trjs_train = self.ObsAct_trajectories[:NTrain]
        trjs_val = self.ObsAct_trajectories[NTrain:]

        losses_train = []
        losses_val = []

        self.theta = self.theta.detach().requires_grad_(False)
        self.Tmat = fun.torch_softmax_2dim(self.theta, dims = (2, 3)).detach().requires_grad_(False)

        print(f"Training with {NTrain} trajectories and validating with {NVal} trajectories.")

        best_val_loss = float('inf')
        best_epoch = -1
        best_params = None


        for epoch in range(NEpochs):
            running_loss = 0.0
            random.shuffle(trjs_train)

            for idx in range(0, NTrain, NBatch):
                self.optimizer.zero_grad()
                loss = torch.tensor(0.0, requires_grad = True)

                if self.minus1_opt:
                    self.psi = torch.concatenate([torch.ones(1, device = self.device), self.psi_toopt],
                                                dim = 0)

                self.rho = nn.Softmax(dim = 0)(self.psi)
                self.TMat = fun.torch_softmax_2dim(self.theta, dims = (2, 3))

                count = 0
                for idx_traj in range(idx, idx + NBatch):
                    if idx_traj < NTrain:
                        # here we go without the custom space for simplicity
                        loss_traj = self.loss(trjs_train[idx_traj][0], trjs_train[idx_traj][1])
                        loss = loss + loss_traj
                        count += 1

                #loss += lam/count*torch.sqrt(self.psi.pow(2).sum())
                
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()/count

                #self.psi.data /= torch.norm(self.psi.data, p = 2)
                #self.psi.data = nn.functional.normalize(self.psi.data, p = 2, dim = 0)

                if verbose:
                    print(f"\t Epoch {epoch + 1} - Batch {idx//NBatch + 1} - Loss: {loss.item()/count}")
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

            if running_loss_val < best_val_loss:
                best_val_loss = running_loss_val
                best_epoch = epoch
                best_params = self.psi.clone()

            print(f"Epoch {epoch + 1} - Training loss: {running_loss}, Validation loss: {running_loss_val} - Learning rate: {self.optimizer.param_groups[0]['lr']}")

            scheduler.step()

        self.trained = True
        self.best_val_loss = best_val_loss
        self.best_epoch = best_epoch
        self.best_params = best_params

        print(f"Best validation loss: {best_val_loss} reached at epoch {best_epoch + 1}")

        return losses_train, losses_val

    def optimize_thetaonly(self, NEpochs, NBatch, lr, train_split = 0.8, optimizer = None, gamma = 0.9):
        """
        Method to optimize the transition probabilities of the FSC using the loaded trajectories, while keeping
        the initial memory occupation fixed. The optimization is performed using the Adam optimizer with a learning
        rate schedule. The trajectories are split into a training and a validation set, and the loss is computed for
        both sets at each epoch. The training loss is computed over random batches of trajectories.
        
        Parameters:
        --- NEpochs: int
            Number of epochs for the optimization.
        --- NBatch: int
            Number of trajectories per batch.
        --- lr: float
            Initial learning rate for the optimizer.
        --- train_split: float (default = 0.8)
            Fraction of trajectories to use for training.
        --- optimizer: torch.optim (default = None)
            Optimizer to use for the optimization. If None, the Adam optimizer is used.
        --- gamma: float (default = 0.9)
            Decay factor for the learning rate schedule.

        Returns:
        --- losses_train: list
            List of training losses at each epoch.
        --- losses_val: list
            List of validation losses at each epoch.
        """
        assert self.trajectories_loaded, "No trajectories have been loaded. Load trajectories with the load_trajectories method."
        assert self.trained == False, "The model has already been trained. If you want to train it again, reinitialize it or set the flaf self.trained to False."

        if optimizer is not None:
            self.optimizer = optimizer([self.theta], lr = lr)
        else:
            self.optimizer = torch.optim.Adam([self.theta], lr = lr)

        # add an exponential scheduler
        scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = gamma)

        NTrain = int(train_split * len(self.ObsAct_trajectories))
        NVal = len(self.ObsAct_trajectories) - NTrain

        trjs_train = self.ObsAct_trajectories[:NTrain]
        trjs_val = self.ObsAct_trajectories[NTrain:]

        losses_train = []
        losses_val = []

        self.psi = self.psi.detach().requires_grad_(False)
        self.rho = nn.Softmax(dim = 0)(self.psi).detach().requires_grad_(False)

        print(f"Training with {NTrain} trajectories and validating with {NVal} trajectories.")

        for epoch in range(NEpochs):
            running_loss = 0.0
            random.shuffle(trjs_train)

            for idx in range(0, NTrain, NBatch):
                self.optimizer.zero_grad()
                loss = torch.tensor(0.0, requires_grad = True)

                self.rho = nn.Softmax(dim = 0)(self.psi)
                self.TMat = fun.torch_softmax_2dim(self.theta, dims = (2, 3))

                count = 0
                for idx_traj in range(idx, idx + NBatch):
                    if idx_traj < NTrain:
                        # here we go without the custom space for simplicity
                        loss_traj = self.loss(trjs_train[idx_traj][0], trjs_train[idx_traj][1])
                        loss = loss + loss_traj
                        count += 1

                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()/count

                print(f"\t Epoch {epoch + 1} - Batch {idx//NBatch + 1} - Loss: {loss.item()/count} - Learning rate: {self.optimizer.param_groups[0]['lr']}")
            running_loss = running_loss/(NTrain//NBatch)

            losses_train.append(running_loss)

            scheduler.step()

            running_loss_val = 0.0

            for idx_traj in range(NVal):
                loss_val = torch.tensor(0.0, requires_grad = False)

                loss_traj_val = self.loss(trjs_val[idx_traj][0], trjs_val[idx_traj][1], grad_required = False)
                loss_val = loss_val + loss_traj_val

                running_loss_val += loss_val.item()

            running_loss_val = running_loss_val/NVal

            losses_val.append(running_loss_val)

            print(f"Epoch {epoch + 1} - Training loss: {running_loss}, Validation loss: {running_loss_val}")

        self.trained = True

        return losses_train, losses_val


    def load_theta(self, theta):
        """
        Loads a new set of parameters for the transition probabilities of the FSC.

        Parameters:
        --- theta: torch.tensor of shape (Y, M, M, A)
            New parameters for the transition probabilities.
        """
        self.theta = nn.Parameter(torch.tensor(theta, device = self.device))
        self.TMat = fun.torch_softmax_2dim(self.theta, dims = (2, 3))
        self.policy = torch.sum(self.TMat, dim = 2)

    def load_psi(self, psi):
        """
        Loads a new set of parameters for the initial memory occupation of the FSC.

        Parameters:
        --- psi: torch.tensor of shape (M)
            New parameters for the initial memory occupation.
        """
        self.psi = nn.Parameter(torch.tensor(psi, device = self.device))
        self.rho = nn.Softmax(dim = 0)(self.psi)

