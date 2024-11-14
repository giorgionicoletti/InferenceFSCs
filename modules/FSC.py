import numpy as np
import numba as nb
import fun
import torch
from torch import nn
import random



class GenerationDiscreteObs():

    def __init__(self, theta, psi, verbose = False,
                 ObsSpace = None, ActSpace = None, MemSpace = None):
        """
        Parameters:
        --- theta: np.array of shape (Y, M, M, A)
            Parameters for the FSC transition probability.
            We assume that the first dimension is the number of observations, the second
            dimention is the number of (initial) memory states, the third dimension is the
            number of (final) memory states, and the last dimension is the number of actions.
        --- psi: np.array of shape (M)
            Parameters for the initial memory occupation.

        By initializing the FSC, the initial memory occupation is computed as rho = softmax(psi),
        and the transition probability is computed as T = softmax(theta, axis = (2, 3)), so that
        it is the conditional probability of taking an action and transitioning to a new memory state
        given the current memory state and the observation.
        """
        self.theta = theta
        self.psi = psi

        self.M = psi.size
        self.A = theta.shape[3]
        self.Y = theta.shape[0]

        if verbose:
            print(f"Initializing FSC with {self.M} memory states, {self.A} actions, and {self.Y} observations.")

        assert self.M == theta.shape[1]
        assert self.M == theta.shape[2]

        if ObsSpace is not None:
            assert len(ObsSpace) == self.Y, "The number of observations in ObsSpace must match the number of observations in theta."
            self.ObsSpace = np.array(ObsSpace)
            self.custom_obs_space = True
        else:
            self.ObsSpace = np.arange(self.Y)
            self.custom_obs_space = False

        if ActSpace is not None:
            assert len(ActSpace) == self.A, "The number of actions in ActSpace must match the number of actions in theta."
            self.ActSpace = np.array(ActSpace)
            self.custom_act_space = True
        else:
            self.ActSpace = np.arange(self.A)
            self.custom_act_space = False

        if MemSpace is not None:
            assert len(MemSpace) == self.M, "The number of memory states in MemSpace must match the number of memory states in theta."
            self.MemSpace = np.array(MemSpace)
            self.custom_mem_space = True
        else:
            self.MemSpace = np.arange(self.M)
            self.custom_mem_space = False

        self.rho = fun.softmax(psi)
        self.TMat = fun.softmax(theta, axis = (2, 3))
        self.policy = np.sum(self.TMat, axis = 2)

    def load_observations(self, observations):
        """
        Loads a sequence of observations to be used to generate a trajectory.

        Parameters:
        --- observations: np.array of shape (NSteps)
            Sequence of observations to be used to generate a trajectory.
        """

        assert np.all(observations < self.Y)
        assert np.all(observations >= 0)

        self.observations = observations
        self.observations_lengths = np.array([len(obs) for obs in observations])
        self.min_obs_length = np.min(self.observations_lengths)
        self.max_obs_length = np.max(self.observations_lengths)

    def generate_single_trajectory(self, NSteps, observations = None, idx_observation = None):
        if observations is None:
            assert hasattr(self, "observations"), "No observations have been loaded. Load observations with the load_observations method."
            assert idx_observation is not None, "If no observations are provided, the idx_observation parameter must not be None."
            assert NSteps <= self.observations[idx_observation].size, "NSteps must be smaller or equal than the number of observations."

            observations_cut = self.observations[idx_observation][:NSteps]
        else:
            assert NSteps <= observations.size, "NSteps must be smaller or equal than the number of observations."
            observations_cut = observations[:NSteps]

        MASpace = fun.combine_spaces(self.MemSpace, self.ActSpace)
        observations_cut = self.observations[:NSteps]

        actions, memories = GenerationDiscreteObs._nb_generate_trajectory(NSteps, self.MemSpace, MASpace, self.TMat, self.rho, observations_cut)
        trajectory = {"actions": actions, "memories": memories, "observations": observations_cut}

        return trajectory
    
    def generate_trajectories(self, NSteps, observations = None, idx_observation = None, NTraj = None,
                              verbose = False):
        if observations is None:
            if idx_observation is None:
                if verbose:
                    print("No observations provided. Using the loaded observations and generating one trajectory per observation sequence.")
                assert hasattr(self, "observations"), "No observations have been loaded. Load observations with the load_observations method."
                assert NSteps <= self.min_obs_length, "NSteps must be smaller than the minimum observation length."
                NTraj = len(self.observations)
                observations_cut = np.zeros((NTraj, NSteps), dtype = np.int32)

                for n in range(NTraj):
                    observations_cut[n] = self.observations[n][:NSteps]
            
            else:
                if verbose:
                    print("No observations provided. Using the indexed observation sequence and generating NTraj trajectories for the same observation sequence.")
                assert hasattr(self, "observations"), "No observations have been loaded. Load observations with the load_observations method."
                assert NSteps <= self.observations[idx_observation].size, "NSteps must be smaller or equal than the number of observations."
                assert NTraj is not None, "If no observations are provided, the NTraj parameter must be provided."
                observations_cut = np.zeros((NTraj, NSteps), dtype = np.int32)

                for n in range(NTraj):
                    observations_cut[n] = self.observations[idx_observation][:NSteps]

        elif type(observations[0]) is np.ndarray:
            if verbose:
                print("Multiple observation sequences provided. Generating one trajectory per observation sequence.")
            obs_lengths = np.array([len(obs) for obs in observations])
            assert np.all(obs_lengths >= NSteps), "All observation sequences must have at least NSteps observations."
            observations_cut = np.zeros((len(observations), NSteps), dtype = np.int32)

            for n in range(len(observations)):
                observations_cut[n] = observations[n][:NSteps]

            NTraj = len(observations)

        else:
            if verbose:
                print("Single observation sequence provided. Generating NTraj trajectories for the same observation sequence.")
            assert NSteps <= observations.size, "NSteps must be smaller or equal than the number of observations."
            assert NTraj is not None, "If observations is a single array, the NTraj parameter must be provided."
            observations_cut = np.zeros((NTraj, NSteps), dtype = np.int32)

            for n in range(NTraj):
                observations_cut[n] = observations[:NSteps]
        
        MASpace = fun.combine_spaces(self.MemSpace, self.ActSpace)

        actions, memories = GenerationDiscreteObs._nb_generate_trajectories_parallel(NTraj, NSteps, self.MemSpace, MASpace,
                                                                                     self.TMat, self.rho, observations_cut)
        trajectories = []

        for n in range(NTraj):
            trajectory = {"actions": actions[n], "memories": memories[n], "observations": observations_cut[n]}
            trajectories.append(trajectory)
        
        return trajectories
    
    def evaluate_nloglikelihood(self, trajectory):

        actions = trajectory["actions"]
        observations = trajectory["observations"]

        nLL = GenerationDiscreteObs._nb_evaluate_nloglikelihood(observations, actions, self.TMat, self.rho, self.ActSpace, self.MemSpace,
                                                                loaded_act_space = self.custom_act_space,
                                                                loaded_obs_space = self.custom_obs_space)

        return nLL
        
    def get_obs_idx(self, obs):
        assert obs in self.ObsSpace, f"Observation {obs} is not in the observation space."
        return np.where(self.ObsSpace == obs)[0][0]
    
    def get_act_idx(self, act):
        assert act in self.ActSpace, f"Action {act} is not in the action space."
        return np.where(self.ActSpace == act)[0][0]
    
    def get_mem_idx(self, mem):
        assert mem in self.MemSpace, f"Memory state {mem} is not in the memory space."
        return np.where(self.MemSpace == mem)[0][0]
    

    @staticmethod
    @nb.njit
    def _nb_evaluate_nloglikelihood(observations, actions, TMat, rho, ActSpace, MemSpace,
                                    loaded_act_space = False, loaded_obs_space = False):
            nLL = 0.
    
            for t, obs in enumerate(observations):
                a = actions[t]
                if loaded_act_space:
                    idx_a = np.where(ActSpace == a)[0][0]
                else:
                    idx_a = a

                if loaded_obs_space:
                    idx_obs = np.where(MemSpace == obs)[0][0]
                else:
                    idx_obs = obs
    
                transition_probs = TMat[idx_obs, :, :, idx_a].T
    
                if t == 0:
                    m = transition_probs @ rho
                else:
                    m = transition_probs @ m
    
                mv = np.sum(m)
                nLL = nLL - np.log(mv)
                m /= mv
    
            return nLL - np.log(np.sum(m))

    @staticmethod
    @nb.njit
    def _nb_generate_trajectory(NSteps, MSpace, MASpace, TMat, rho, observations):

        actions = np.zeros(NSteps, dtype = np.int32)
        memories = np.zeros(NSteps, dtype = np.int32)

        initial_memory = fun.numba_random_choice(MSpace, rho)

        memories[0] = initial_memory

        for t in range(1, NSteps):
            transition_probs = TMat[observations[t-1], memories[t-1]].flatten()
            new_MA = fun.numba_random_choice(MASpace, transition_probs)
            memories[t] = new_MA[0]
            actions[t-1] = new_MA[1]

        return actions, memories
    
    @staticmethod
    @nb.njit
    def _nb_generate_trajectories_parallel(NTraj, NSteps, MSpace, MASpace, TMat, rho, observations):
            
            actions = np.zeros((NTraj, NSteps), dtype = np.int32)
            memories = np.zeros((NTraj, NSteps), dtype = np.int32)
    
            for n in nb.prange(NTraj):
                initial_memory = fun.numba_random_choice(MSpace, rho)
    
                memories[n, 0] = initial_memory
                actions[n, 0] = -1
    
                for t in range(1, NSteps):
                    transition_probs = TMat[observations[n, t-1], memories[n, t-1]].flatten()
                    new_MA = fun.numba_random_choice(MASpace, transition_probs)
                    memories[n, t] = new_MA[0]
                    actions[n, t -1] = new_MA[1]
    
            return actions, memories

    

class InferenceDiscreteObs():

    def __init__(self, M, A, Y,
                 ObsSpace = None, ActSpace = None, MemSpace = None,
                 seed = None):

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
        self.psi = nn.Parameter(torch.randn(self.M, device = self.device))

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
        self.ObsAct_trajectories = []

        for trajectory in trajectories:
            self.ObsAct_trajectories.append([torch.tensor(trajectory["observations"]),
                                             torch.tensor(trajectory["actions"])])
            
        self.trajectories_loaded = True

    def get_obs_idx(self, obs):
        if self.custom_obs_space:
            assert obs in self.ObsSpace, f"Observation {obs} is not in the observation space."
            return torch.where(self.ObsSpace == obs)[0][0]
        else:
            return obs
    
    def get_act_idx(self, act):
        if self.custom_act_space:
            assert act in self.ActSpace, f"Action {act} is not in the action space."
            return torch.where(self.ActSpace == act)[0][0]
        else:
            return act

    def evaluate_nloglikelihood(self, idx_traj, grad_required = False):
        observations, actions = self.ObsAct_trajectories[idx_traj]
        if self.custom_obs_space or self.custom_act_space:
            return self.loss_customspace(observations, actions, grad_required = grad_required)
        else:
            return self.loss(observations, actions, grad_required = grad_required)
        
    def loss_customspace(self, observations, actions, grad_required = True):
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


    def optimize(self, NEpochs, NBatch, lr, train_split = 0.8, optimizer = None):
        assert self.trajectories_loaded, "No trajectories have been loaded. Load trajectories with the load_trajectories method."
        assert self.trained == False, "The model has already been trained. If you want to train it again, reinitialize it or set the flaf self.trained to False."

        if optimizer is not None:
            self.optimizer = optimizer([self.theta, self.psi], lr = lr)
        else:
            self.optimizer = torch.optim.Adam([self.theta, self.psi], lr = lr)

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

                for idx_traj in range(idx, idx + NBatch):
                    if idx_traj < NTrain:
                        # here we go without the custom space for simplicity
                        loss_traj = self.loss(trjs_train[idx_traj][0], trjs_train[idx_traj][1])
                        loss = loss + loss_traj

                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()/NBatch

                print(f"\t Epoch {epoch + 1} - Batch {idx//NBatch + 1} - Loss: {loss.item()}")
            losses_train.append(running_loss)

            running_loss_val = 0.0

            for idx in range(0, NVal, NBatch):
                loss_val = torch.tensor(0.0, requires_grad = False)

                for idx_traj in range(idx, idx + NBatch):
                    if idx_traj < NVal:
                        loss_traj_val = self.loss(trjs_val[idx_traj][0], trjs_val[idx_traj][1], grad_required = False)
                        loss_val = loss_val + loss_traj_val

                running_loss_val += loss_val.item()

            losses_val.append(running_loss_val)

            print(f"Epoch {epoch + 1} - Training loss: {running_loss}, Validation loss: {running_loss_val}")

        self.trained = True

        return losses_train, losses_val

