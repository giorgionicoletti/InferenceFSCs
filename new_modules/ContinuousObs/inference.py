import torch
from torch import nn
import numpy as np
import numba as nb
import random

import utils

class InferenceContinuousObs:
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

        self.InternalMemSpace = torch.arange(self.FSC.M)
        self.InternalActSpace = torch.arange(self.FSC.A)

        self.trajectories_loaded = False
        self.optimizer_initialized = False
        self.trained = False

    def get_TMat(self, f):
        if len(f.shape) == 1:
            W = torch.einsum('fmna, f -> mna', self.theta, f)
            TM = utils.torch_softmax_2dim(W, dims=(1, 2))
        else:
            W = torch.einsum('fmna, fz -> zmna', self.theta, f)
            TM = utils.torch_softmax_2dim(W, dims=(2, 3))
        return TM

    def load_trajectories(self, trajectories):
        """
        Loads a set of trajectories to be used for training the FSC.

        Parameters:
        --- trajectories: list of dicts
            List of dictionaries containing the actions and features for each trajectory.
        """
        self.FeatAct_trajectories = []

        for trajectory in trajectories:
            features = torch.tensor(trajectory["features"].astype(np.float32)).to(self.device)
            actions = self.__map_act_to_internal_space(trajectory["actions"])
            actions = torch.tensor(actions.astype(np.int32)).to(self.device)
            self.FeatAct_trajectories.append([features, actions])

        self.trajectories_loaded = True

    def evaluate_nloglikelihood(self, idx_traj, grad_required=False):
        """
        Wrapper method to evaluate the negative log-likelihood of a given trajectory.

        Parameters:
        --- idx_traj: int
            Index of the trajectory to evaluate.
        --- grad_required: bool (default = False)
            Flag indicating whether the gradient is required or not.

        Returns:
        --- nLL: float
            Negative log-likelihood of the trajectory.
        """
        features, actions = self.FeatAct_trajectories[idx_traj]
        return self.loss(features, actions, grad_required=grad_required)

    def loss(self, features, actions, grad_required=True):
        """
        Method to compute the negative log-likelihood of a given trajectory.
        The gradients of the loss are computed if the grad_required flag is set to True.

        Parameters:
        --- features: torch.tensor
            Array of features.
        --- actions: torch.tensor
            Array of actions.
        --- grad_required: bool (default = True)
            Flag indicating whether the gradient is required or not.

        Returns:
        --- nLL: float
            Negative log-likelihood of the trajectory.
        """
        nLL = torch.tensor(0.0, requires_grad=grad_required)

        for t in range(features.size(1)):
            idx_a = actions[t]
            TMat = self.get_TMat(features[:, t])

            transition_probs = TMat[:, :, idx_a].T

            if t == 0:
                m = torch.matmul(transition_probs, self.rho)
            else:
                m = torch.matmul(transition_probs, m)

            mv = torch.sum(m)
            nLL = nLL - torch.log(mv)
            m /= mv

        return nLL - torch.log(torch.sum(m))

    def optimize(self, NEpochs, NBatch, lr_theta, lr_psi, train_split=0.8, optimizer=None, gamma=0.9, verbose=False):
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
        --- lr_theta: float
            Initial learning rate for the optimizer for theta.
        --- lr_psi: float
            Initial learning rate for the optimizer for psi.
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
        assert not self.trained, "The model has already been trained. If you want to train it again, reinitialize it or set the flag self.trained to False."

        if optimizer is not None:
            self.optimizer = optimizer([{'params': self.theta, 'lr': lr_theta}, {'params': self.psi, 'lr': lr_psi}])
        else:
            self.optimizer = torch.optim.Adam([{'params': self.theta, 'lr': lr_theta}, {'params': self.psi, 'lr': lr_psi}])

        scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)

        NTrain = int(train_split * len(self.FeatAct_trajectories))
        NVal = len(self.FeatAct_trajectories) - NTrain

        trjs_train = self.FeatAct_trajectories[:NTrain]
        trjs_val = self.FeatAct_trajectories[NTrain:]

        losses_train = []
        losses_val = []

        print(f"Training with {NTrain} trajectories and validating with {NVal} trajectories.")

        for epoch in range(NEpochs):
            running_loss = 0.0
            random.shuffle(trjs_train)

            for idx in range(0, NTrain, NBatch):
                self.optimizer.zero_grad()
                loss = torch.tensor(0.0, requires_grad=True)

                self.rho = nn.Softmax(dim=0)(self.psi)

                count = 0

                for idx_traj in range(idx, idx + NBatch):
                    if idx_traj < NTrain:
                        loss_traj = self.loss(trjs_train[idx_traj][0], trjs_train[idx_traj][1])
                        loss = loss + loss_traj
                        count += 1

                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() / count

                if verbose:
                    print(f"\t Epoch {epoch + 1} - Batch {idx // NBatch + 1} - Loss: {loss.item() / count} - Learning rate: {self.optimizer.param_groups[0]['lr']}")

            running_loss = running_loss / (NTrain // NBatch)
            losses_train.append(running_loss)

            running_loss_val = 0.0

            for idx_traj in range(NVal):
                loss_val = torch.tensor(0.0, requires_grad=False)

                loss_traj_val = self.loss(trjs_val[idx_traj][0], trjs_val[idx_traj][1], grad_required=False)
                loss_val = loss_val + loss_traj_val

                running_loss_val += loss_val.item()

            running_loss_val = running_loss_val / NVal
            losses_val.append(running_loss_val)

            print(f"Epoch {epoch + 1} - Training loss: {running_loss}, Validation loss: {running_loss_val} - Learning rate: {self.optimizer.param_groups[0]['lr']}")

            scheduler.step()

        self.trained = True

        return losses_train, losses_val
    

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
    # def load_theta(self, theta):
    #     """
    #     Loads a new set of parameters for the transition probabilities of the FSC.

    #     Parameters:
    #     --- theta: torch.tensor of shape (F, M, M, A)
    #         New parameters for the transition probabilities.
    #     """
    #     self.theta = nn.Parameter(torch.tensor(theta, device=self.device))

    # def load_psi(self, psi):
    #     """
    #     Loads a new set of parameters for the initial memory occupation of the FSC.

    #     Parameters:
    #     --- psi: torch.tensor of shape (M)
    #         New parameters for the initial memory occupation.
    #     """
    #     self.psi = nn.Parameter(torch.tensor(psi, device=self.device))
    #     self.rho = nn.Softmax(dim=0)(self.psi)


    def optimize_double(self, NEpochs, NBatch, lr_theta, lr_psi, train_split=0.8, optimizer=None, gamma=0.9, verbose=False):
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
        --- lr_theta: float
            Initial learning rate for the optimizer for theta.
        --- lr_psi: float
            Initial learning rate for the optimizer for psi.
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
        assert not self.trained, "The model has already been trained. If you want to train it again, reinitialize it or set the flag self.trained to False."

        optimizer_theta = torch.optim.Adam([self.theta], lr=lr_theta)
        

        scheduler_theta = torch.optim.lr_scheduler.ExponentialLR(optimizer_theta, gamma=gamma)

        NTrain = int(train_split * len(self.FeatAct_trajectories))
        NVal = len(self.FeatAct_trajectories) - NTrain

        trjs_train = self.FeatAct_trajectories[:NTrain]
        trjs_val = self.FeatAct_trajectories[NTrain:]

        losses_train = []
        losses_val = []

        print(f"Training with {NTrain} trajectories and validating with {NVal} trajectories.")

        for epoch in range(NEpochs):
            running_loss = 0.0
            random.shuffle(trjs_train)

            # Optimize psi
            optimizer_psi = torch.optim.Adam([self.psi], lr=lr_psi)
            scheduler_psi = torch.optim.lr_scheduler.ExponentialLR(optimizer_psi, gamma=0.9)
            for _ in range(50):
                for idx in range(0, NTrain, NBatch):
                    optimizer_psi.zero_grad()
                    loss = torch.tensor(0.0, requires_grad=True)
                    self.rho = nn.Softmax(dim=0)(self.psi)

                    for idx_traj in range(idx, idx + NBatch):
                        if idx_traj < NTrain:
                            features, actions = trjs_train[idx_traj]
                            TMat = self.get_TMat(features[:, 0])
                            transition_probs = TMat[:, :, actions[0]].T
                            m = torch.matmul(transition_probs, self.rho)
                            mv = torch.sum(m)
                            loss = loss - torch.log(mv)
                            m /= mv

                    loss.backward()
                    optimizer_psi.step()

                scheduler_psi.step()

            # Optimize theta
            for idx in range(0, NTrain, NBatch):
                optimizer_theta.zero_grad()
                loss = torch.tensor(0.0, requires_grad=True)

                self.rho = nn.Softmax(dim=0)(self.psi)

                count = 0

                for idx_traj in range(idx, idx + NBatch):
                    if idx_traj < NTrain:
                        loss_traj = self.loss(trjs_train[idx_traj][0], trjs_train[idx_traj][1])
                        loss = loss + loss_traj
                        count += 1

                loss.backward()
                optimizer_theta.step()
                running_loss += loss.item() / count

                if verbose:
                    print(f"\t Epoch {epoch + 1} - Batch {idx // NBatch + 1} - Loss: {loss.item() / count} - Learning rate: {optimizer_theta.param_groups[0]['lr']}")

            running_loss = running_loss / (NTrain // NBatch)
            losses_train.append(running_loss)

            running_loss_val = 0.0

            for idx_traj in range(NVal):
                loss_val = torch.tensor(0.0, requires_grad=False)

                loss_traj_val = self.loss(trjs_val[idx_traj][0], trjs_val[idx_traj][1], grad_required=False)
                loss_val = loss_val + loss_traj_val

                running_loss_val += loss_val.item()

            running_loss_val = running_loss_val / NVal
            losses_val.append(running_loss_val)

            print(f"Epoch {epoch + 1} - Training loss: {running_loss}, Validation loss: {running_loss_val} - Learning rate: {optimizer_theta.param_groups[0]['lr']}")

            scheduler_theta.step()
            

        self.trained = True

        return losses_train, losses_val