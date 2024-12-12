import numpy as np
import numba as nb
import fun
import torch
from torch import nn
import random

import matplotlib.pyplot as plt
import scipy

class GenerationDiscreteObs():

    def __init__(self, theta, psi, verbose = False,
                 ObsSpace = None, ActSpace = None, MemSpace = None):
        """
        Creates an instance of a Finite State Controller (FSC) for discrete observations. The parameters
        theta and psi are used to define the transition probabilities and the initial memory occupation,
        respectively. In particular, the initial memory occupation is computed as rho = softmax(psi),
        and the transition probability is computed as T = softmax(theta, axis = (2, 3)), so that
        it is the conditional probability of taking an action and transitioning to a new memory state
        given the current memory state and the observation.

        This class is only able to generate trajectories and evaluate the negative log-likelihood of a
        given trajectory. For training, use the InferenceDiscreteObs class for the moment.

        Parameters:
        --- theta: np.array of shape (Y, M, M, A)
            Parameters for the FSC transition probability.
            We assume that the first dimension is the number of observations, the second
            dimention is the number of (initial) memory states, the third dimension is the
            number of (final) memory states, and the last dimension is the number of actions.
        --- psi: np.array of shape (M)
            Parameters for the initial memory occupation.
        --- verbose: bool (default = False)
            If True, prints information about the FSC initialization.
        --- ObsSpace: list (default = None)
            List of observations. If None, the observations are assumed to be integers from 0 to Y-1.
        --- ActSpace: list (default = None)
            List of actions. If None, the actions are assumed to be integers from 0 to A-1.
        --- MemSpace: list (default = None)
            List of memory states. If None, the memory states are assumed to be integers from 0 to M-1.
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
        --- observations: list of np.arrays
            List of observation sequences. Each element of the list is an np.array of possibly different lengths,
            containing the observations for a single trajectory.
        """

        assert np.all(observations < self.Y)
        assert np.all(observations >= 0)

        self.observations = observations
        self.observations_lengths = np.array([len(obs) for obs in observations])
        self.min_obs_length = np.min(self.observations_lengths)
        self.max_obs_length = np.max(self.observations_lengths)

    def generate_single_trajectory(self, NSteps, observations = None, idx_observation = None):
        """
        Generates a single trajectory of NSteps length given a sequence of observations. If no observations are provided,
        the method uses the loaded observations and generates a trajectory for the indexed observation sequence.

        In any case, the number of steps NSteps must be smaller or equal than the number of observations in the provided
        observation sequence.

        Note that the trajectory is not stored in the object, but returned as a dictionary.

        Parameters:
        --- NSteps: int
            Number of steps for the trajectory.
        --- observations: np.array (default = None)
            Array of observations. If None, the method uses the loaded observations.
        --- idx_observation: int (default = None)
            Index of the observation sequence to use if no observations are provided.

        Returns:
        --- trajectory: dict
            Dictionary containing the actions, memories, and observations for the generated trajectory.
        """
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
        """
        Generates NTraj trajectories of NSteps length given a sequence of observations. If no observations are provided,
        the method uses the loaded observations. It is also possible to generate NTraj trajectories for the same observation
        sequence by providing the idx_observation parameter and setting the NTraj parameter.

        In any case, the number of steps NSteps must be smaller or equal than the number of observations in the provided
        observation sequence.

        Note that the trajectories are not stored in the object, but returned as a list of dictionaries.

        Parameters:
        --- NSteps: int
            Number of steps for the trajectory.
        --- observations: list of np.arrays (default = None)
            List of observation sequences. If None, the method uses the loaded observations.
        --- idx_observation: int (default = None)
            Index of the observation sequence to use if no observations are provided.
        --- NTraj: int (default = None)
            Number of trajectories to generate. If None, the method generates one trajectory per observation sequence.
        --- verbose: bool (default = False)
            If True, prints information about the generation process.

        Returns:
        --- trajectories: list of dicts
            List of dictionaries containing the actions, memories, and observations for each generated trajectory.
        """
        if observations is None:
            if idx_observation is None:
                if verbose:
                    print("No observations provided. Using the loaded observations and generating one trajectory per observation sequence.")
                assert hasattr(self, "observations"), "No observations have been loaded. Load observations with the load_observations method."
                assert NSteps <= self.min_obs_length, "NSteps must be smaller than the shortest observation length."
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
        """
        Evaluates the negative log-likelihood of a given trajectory.

        Parameters:
        --- trajectory: dict
            Dictionary containing the actions, memories, and observations for the trajectory.

        Returns:
        --- nLL: float
            Negative log-likelihood of the trajectory.
        """

        actions = trajectory["actions"]
        observations = trajectory["observations"]

        nLL = GenerationDiscreteObs._nb_evaluate_nloglikelihood(observations, actions, self.TMat, self.rho, self.ActSpace, self.MemSpace,
                                                                loaded_act_space = self.custom_act_space,
                                                                loaded_obs_space = self.custom_obs_space)

        return nLL
        
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
        assert obs in self.ObsSpace, f"Observation {obs} is not in the observation space."
        return np.where(self.ObsSpace == obs)[0][0]
    
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
        assert act in self.ActSpace, f"Action {act} is not in the action space."
        return np.where(self.ActSpace == act)[0][0]
    
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
        assert mem in self.MemSpace, f"Memory state {mem} is not in the memory space."
        return np.where(self.MemSpace == mem)[0][0]

    def plot_trajectory(self, trj, Time):
        """
        Plots the actions, memories, and observations of a given trajectory.

        Parameters:
        --- trj: dict
            Dictionary containing the actions, memories, and observations for the trajectory.
        --- Time: np.array
            Array of time steps for the trajectory.

        Returns:
        --- fig: plt.figure
            Figure object containing the plots.
        --- ax: np.array of plt.axes
            Array of axes objects containing the plots.
        """
        fig, ax = plt.subplots(3,1, figsize=(10,5))
        plt.subplots_adjust(hspace=0.5)

        ax[0].plot(Time, trj["observations"], 'o', c= 'k')
        ax[0].plot(Time, trj["observations"], c = 'k')
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('Observations')

        ax[1].plot(Time, trj["memories"], 'o', c= 'k')
        ax[1].plot(Time, trj["memories"], c = 'k')
        ax[1].set_xlabel('Time')
        ax[1].set_ylabel('Memories')

        ax[2].plot(Time, trj["actions"], 'o', c= 'k')
        ax[2].plot(Time, trj["actions"], c = 'k')
        ax[2].set_xlabel('Time')
        ax[2].set_ylabel('Actions')

        return fig, ax
    
    def load_theta(self, theta):
        """
        Loads a new set of parameters for the transition probabilities of the FSC.

        Parameters:
        --- theta: np.array of shape (Y, M, M, A)
            New parameters for the transition probabilities.
        """
        self.theta = theta
        self.TMat = fun.softmax(theta, axis = (2, 3))
        self.policy = np.sum(self.TMat, axis = 2)

    def load_psi(self, psi):
        """
        Loads a new set of parameters for the initial memory occupation of the FSC.

        Parameters:
        --- psi: np.array of shape (M)
            New parameters for the initial memory occupation.
        """
        self.psi = psi
        self.rho = fun.softmax(psi)
    

    @staticmethod
    @nb.njit
    def _nb_evaluate_nloglikelihood(observations, actions, TMat, rho, ActSpace, MemSpace,
                                    loaded_act_space = False, loaded_obs_space = False):
            """
            Static method providing a numba-compiled implementation of the negative log-likelihood evaluation
            for a given trajectory. The method is then wrapped in the evaluate_nloglikelihood method.

            Parameters:
            --- observations: np.array
                Array of observations.
            --- actions: np.array
                Array of actions.
            --- TMat: np.array of shape (Y, M, M, A)
                Transition probability matrix.
            --- rho: np.array of shape (M)
                Initial memory occupation.
            --- ActSpace: np.array
                Array of actions.
            --- MemSpace: np.array
                Array of memory states.
            --- loaded_act_space: bool (default = False)
                Flag indicating whether the action space is custom or not.
                If not, the actions are assumed to be integers from 0 to A-1.
            --- loaded_obs_space: bool (default = False)
                Flag indicating whether the observation space is custom or not.
                If not, the observations are assumed to be integers from 0 to Y-1.

            Returns:
            --- nLL: float
                Negative log-likelihood of the trajectory.
            """
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
        """
        Static method providing a numba-compiled implementation of the trajectory generation
        for a given observation sequence. The method is then wrapped in the generate_single_trajectory method.

        Parameters:
        --- NSteps: int
            Number of steps for the trajectory.
        --- MSpace: np.array
            Array of memory states.
        --- MASpace: np.array
            Array of memory-action pairs.
        --- TMat: np.array of shape (Y, M, M, A)
            Transition probability matrix.
        --- rho: np.array of shape (M)
            Initial memory occupation.
        --- observations: np.array
            Array of observations.

        Returns:
        --- actions: np.array
            Array of actions.
        --- memories: np.array
            Array of memory states.
        """
        actions = np.zeros(NSteps, dtype = np.int32)
        memories = np.zeros(NSteps, dtype = np.int32)

        initial_memory = fun.numba_random_choice(MSpace, rho)

        memories[0] = initial_memory

        for t in range(0, NSteps):
            transition_probs = TMat[observations[t], memories[t]].flatten()
            new_MA = fun.numba_random_choice(MASpace, transition_probs)
            if t < NSteps - 1:
                memories[t + 1] = new_MA[0]
            actions[t] = new_MA[1]

        return actions, memories
    
    @staticmethod
    @nb.njit
    def _nb_generate_trajectories_parallel(NTraj, NSteps, MSpace, MASpace, TMat, rho, observations):
            """
            Static method providing a numba-compiled implementation of the trajectory generation
            for a given observation sequence. The method is then wrapped in the generate_trajectories method,
            and it generates NTraj trajectories in parallel.

            Parameters:
            --- NTraj: int
                Number of trajectories to generate.
            --- NSteps: int
                Number of steps for the trajectory.
            --- MSpace: np.array
                Array of memory states.
            --- MASpace: np.array
                Array of memory-action pairs.
            --- TMat: np.array of shape (Y, M, M, A)
                Transition probability matrix.
            --- rho: np.array of shape (M)
                Initial memory occupation.
            --- observations: np.array
                Array of observations.

            Returns:
            --- actions: np.array
                Array of actions.
            --- memories: np.array
                Array of memory states.
            """
            actions = np.zeros((NTraj, NSteps), dtype = np.int32)
            memories = np.zeros((NTraj, NSteps), dtype = np.int32)
    
            for n in nb.prange(NTraj):
                initial_memory = fun.numba_random_choice(MSpace, rho)
    
                memories[n, 0] = initial_memory
                #actions[n, 0] = -1
    
                for t in range(0, NSteps):
                    transition_probs = TMat[observations[n, t], memories[n, t]].flatten()
                    new_MA = fun.numba_random_choice(MASpace, transition_probs)
                    if t < NSteps - 1:
                        memories[n, t+1] = new_MA[0]
                    actions[n, t] = new_MA[1]
    
            return actions, memories

    

class InferenceDiscreteObs():

    def __init__(self, M, A, Y,
                 ObsSpace = None, ActSpace = None, MemSpace = None,
                 seed = None):
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
        self.psi = torch.randn(self.M, device = self.device)

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

        self.pStart_ya_emp = np.zeros((self.Y, self.A))   

        for trajectory in trajectories:
            self.ObsAct_trajectories.append([torch.tensor(trajectory["observations"]),
                                             torch.tensor(trajectory["actions"])])
            
            y0 = trajectory["observations"][0]
            a0 = trajectory["actions"][0]

            idx_y0 = self.get_obs_idx(y0)
            idx_a0 = self.get_act_idx(a0)

            self.pStart_ya_emp[idx_y0, idx_a0] += 1
        
        self.pStart_ya_emp /= np.sum(self.pStart_ya_emp)
            
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

    def optimize(self, NEpochs, NBatch, lr, train_split = 0.8, optimizer = None, gamma = 0.9,
                 verbose = False, maxiter = 1000, psi0 = None, method = None, hess = None):
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
        assert self.trained == False, "The model has already been trained. If you want to train it again, reinitialize it or set the flag self.trained to False."

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

        if method is None:
            method = 'trust-ncg'

        if hess is None:
            hess = '2-point'


        print(f"Training with {NTrain} trajectories and validating with {NVal} trajectories.")

        for epoch in range(NEpochs):
            running_loss = 0.0
            random.shuffle(trjs_train)

            for idx in range(0, NTrain, NBatch):
                self.optimizer.zero_grad()
                loss = torch.tensor(0.0, requires_grad = True)

                self.TMat = fun.torch_softmax_2dim(self.theta, dims = (2, 3))

                wVec = self.TMat.detach().cpu().numpy().sum(axis = 2).transpose(0, 2, 1)

                if psi0 is None:
                    psi0 = np.zeros(self.M)
                
                res = scipy.optimize.minimize(fun.fun_MSE, psi0, args = (wVec, self.pStart_ya_emp),
                                              jac = fun.jac_MSE, method=method,
                                              hess = hess, options = {'maxiter': maxiter})
                
                self.psi = torch.tensor(res.x.astype(np.float32), device = self.device)
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

    # def optimize_psionly(self, method = None, hess = None, maxiter = 1000, psi0 = None):
    #     assert self.trajectories_loaded, "No trajectories have been loaded. Load trajectories with the load_trajectories method."
    #     assert self.trained == False, "The model has already been trained. If you want to train it again, reinitialize it or set the flag self.trained to False."

    #     if psi0 is None:
    #         psi0 = np.zeros(self.M)

    #     wVec = self.TMat.detach().cpu().numpy().sum(axis = 2).transpose(0, 2, 1)

    #     if method is None:
    #         method = 'trust-ncg'

    #     if hess is None:
    #         hess = '2-point'
        
    #     res = scipy.optimize.minimize(fun.fun_MSE, psi0, args = (wVec, self.pStart_ya_emp),
    #                                   jac = fun.jac_MSE, method=method,
    #                                   hess = hess, options = {'maxiter': maxiter})
        
    #     self.psi = torch.tensor(res.x.astype(np.float32), device = self.device)

    #     self.rho = nn.Softmax(dim = 0)(self.psi)
    #     self.trained = True

    #     return res.fun

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

    def optimize_psionly(self, maxiter = 1000, rho0 = None, th = 1e-6):
        assert self.trajectories_loaded, "No trajectories have been loaded. Load trajectories with the load_trajectories method."
        assert self.trained == False, "The model has already been trained. If you want to train it again, reinitialize it or set the flag self.trained to False."

        if rho0 is None:
            rho0 = np.ones(self.M)/self.M

        rho, err = self.optimize_rho(self.Y, self.M, self.A, self.TMat.detach().cpu().numpy(), self.pStart_ya_emp, rho0, maxiter, th = th)

        self.rho = torch.tensor(rho.astype(np.float32), device = self.device)
        
        self.trained = True

        return err

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















############################################################################################################
############################# FSC for continuous observations ##############################################
############################################################################################################

class GenerationContinuousObs():

    def __init__(self, theta, psi, verbose = False,
                 ActSpace = None, MemSpace = None):
        self.theta = theta
        self.psi = psi

        self.M = psi.size
        self.A = theta.shape[3]
        self.F = theta.shape[0]

        if verbose:
            print(f"Initializing FSC with {self.M} memory states, {self.A} actions, and {self.F} features.")

        assert theta.shape == (self.F, self.M, self.M, self.A), "The shape of theta must be (F, M, M, A)."          

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

    
    def load_features(self, features):
        for ftraj in features:
            assert np.shape(ftraj)[0] == self.F, "The number of features must match the number of features in theta. Check that the second dimension of the features array is equal to F." 
        
        self.features = features
        self.features_lengths = np.array([ftraj.shape[1] for ftraj in features])
        self.min_features_length = np.min(self.features_lengths)
        self.max_features_length = np.max(self.features_lengths)

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
        assert act in self.ActSpace, f"Action {act} is not in the action space."
        return np.where(self.ActSpace == act)[0][0]
    
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
        assert mem in self.MemSpace, f"Memory state {mem} is not in the memory space."
        return np.where(self.MemSpace == mem)[0][0]

    def generate_single_trajectory(self, NSteps, features = None, idx_features = None):
        """
        Generates a single trajectory of NSteps length given a sequence of features. If no features are provided,
        the method uses the loaded features and generates a trajectory for the indexed features sequence.

        In any case, the number of steps NSteps must be smaller or equal than the number of features in the provided
        features sequence.

        Note that the trajectory is not stored in the object, but returned as a dictionary.

        Parameters:
        --- NSteps: int
            Number of steps for the trajectory.
        --- features: np.array (default = None)
            Array of features. If None, the method uses the loaded features.
        --- idx_features: int (default = None)
            Index of the features sequence to use if no features are provided.

        Returns:
        --- trajectory: dict
            Dictionary containing the actions, memories, and features for the generated trajectory.
        """
        if features is None:
            assert hasattr(self, "features"), "No features have been loaded. Load features with the load_features method."
            assert idx_features is not None, "If no features are provided, the idx_features parameter must not be None."
            assert NSteps <= self.features[idx_features].size, "NSteps must be smaller or equal than the number of features."

            features_cut = self.features[idx_features][:, :NSteps]
        else:
            assert NSteps <= features.size, "NSteps must be smaller or equal than the number of features."
            features_cut = features[:, :NSteps]

        MASpace = fun.combine_spaces(self.MemSpace, self.ActSpace)
        features_cut = self.features[:, :NSteps]

        actions, memories = GenerationContinuousObs._nb_generate_trajectory(GenerationContinuousObs._nb_get_TMat,
                                                         NSteps, self.MemSpace, MASpace, self.theta, self.rho, features_cut)
        
        trajectory = {"actions": actions, "memories": memories, "features": features_cut}

        return trajectory
    

    def generate_trajectories(self, NSteps, features = None, idx_features = None, NTraj = None,
                              verbose = False):
        """
        Generates NTraj trajectories of NSteps length given a sequence of features. If no features are provided,
        the method uses the loaded features. It is also possible to generate NTraj trajectories for the same features
        sequence by providing the idx_features parameter and setting the NTraj parameter.

        In any case, the number of steps NSteps must be smaller or equal than the number of features in the provided
        features sequence.

        Note that the trajectories are not stored in the object, but returned as a list of dictionaries.

        Parameters:
        --- NSteps: int
            Number of steps for the trajectory.
        --- features: list of np.arrays (default = None)
            List of features sequences. If None, the method uses the loaded features.
        --- idx_features: int (default = None)
            Index of the features sequence to use if no features are provided.
        --- NTraj: int (default = None)
            Number of trajectories to generate. If None, the method generates one trajectory per features sequence.
        --- verbose: bool (default = False)
            If True, prints information about the generation process.

        Returns:
        --- trajectories: list of dicts
            List of dictionaries containing the actions, memories, and features for each generated trajectory.
        """
        if features is None:
            if idx_features is None:
                if verbose:
                    print("No features provided. Using the loaded features and generating one trajectory per features sequence.")
                assert hasattr(self, "features"), "No features have been loaded. Load features with the load_features method."
                assert NSteps <= self.min_features_length, "NSteps must be smaller than the shortest features length."
                NTraj = len(self.features)
                features_cut = np.zeros((NTraj, self.F, NSteps), dtype = np.float32)

                for n in range(NTraj):
                    features_cut[n] = self.features[n][:, :NSteps]
            
            else:
                if verbose:
                    print("No features provided. Using the indexed features sequence and generating NTraj trajectories for the same features sequence.")
                assert hasattr(self, "features"), "No features have been loaded. Load features with the load_features method."
                assert NSteps <= self.features[idx_features].size, "NSteps must be smaller or equal than the number of features."
                assert NTraj is not None, "If no features are provided, the NTraj parameter must be provided."
                features_cut = np.zeros((NTraj, self.F, NSteps), dtype = np.float32)

                for n in range(NTraj):
                    features_cut[n] = self.features[idx_features][:, :NSteps]

        elif type(features[0]) is np.ndarray:
            if verbose:
                print("Multiple features sequences provided. Generating one trajectory per features sequence.")
            features_lengths = np.array([ftraj.shape[1] for ftraj in features])
            assert np.all(features_lengths >= NSteps), "All features sequences must have at least NSteps features."
            NTraj = len(features)
            features_cut = np.zeros((NTraj, self.F, NSteps), dtype = np.float32)

            for n in range(NTraj):
                features_cut[n] = features[n][:, :NSteps]

        else:
            if verbose:
                print("Single features sequence provided. Generating NTraj trajectories for the same features sequence.")
            assert NSteps <= features.shape[1], "NSteps must be smaller or equal than the number of features."
            assert NTraj is not None, "If features is a single array, the NTraj parameter must be provided."
            features_cut = np.zeros((NTraj, self.F, NSteps), dtype = np.float32)

            for n in range(NTraj):
                features_cut[n] = features[:, :NSteps]
        
        MASpace = fun.combine_spaces(self.MemSpace, self.ActSpace)

        actions, memories = GenerationContinuousObs._nb_generate_trajectories_parallel(GenerationContinuousObs._nb_get_TMat,
                                                                    NTraj, NSteps, self.MemSpace, MASpace,
                                                                    self.theta, self.rho, features_cut)
        trajectories = []

        for n in range(NTraj):
            trajectory = {"actions": actions[n], "memories": memories[n], "features": features_cut[n]}
            trajectories.append(trajectory)
        
        return trajectories


    def plot_trajectory(self, trj, Time, colors_features = None):
        """
        Plots the actions, memories, and features of a given trajectory.

        Parameters:
        --- trj: dict
            Dictionary containing the actions, memories, and features for the trajectory.
        --- Time: np.array
            Array of time steps for the trajectory.

        Returns:
        --- fig: plt.figure
            Figure object containing the plots.
        --- ax: np.array of plt.axes
            Array of axes objects containing the plots.
        """
        fig, ax = plt.subplots(3,1, figsize=(10,5))
        plt.subplots_adjust(hspace=0.5)

        if colors_features is None:
            colors_features = ['k' for i in range(self.F)]

        for f in range(self.F):
            ax[0].plot(Time, trj["features"][f], 'o', c = colors_features[f])
            ax[0].plot(Time, trj["features"][f], c = colors_features[f])
            ax[0].set_xlabel('Time')
            ax[0].set_ylabel('Features')

        ax[1].plot(Time, trj["memories"], 'o', c= 'k')
        ax[1].plot(Time, trj["memories"], c = 'k')
        ax[1].set_xlabel('Time')
        ax[1].set_ylabel('Memories')

        ax[2].plot(Time, trj["actions"], 'o', c= 'k')
        ax[2].plot(Time, trj["actions"], c = 'k')
        ax[2].set_xlabel('Time')
        ax[2].set_ylabel('Actions')

        return fig, ax


    def load_theta(self, theta):
        """
        Loads a new set of parameters for the transition probabilities of the FSC.

        Parameters:
        --- theta: np.array of shape (Y, M, M, A)
            New parameters for the transition probabilities.
        """
        self.theta = theta
        self.TMat = fun.softmax(theta, axis = (2, 3))
        self.policy = np.sum(self.TMat, axis = 2)

    def load_psi(self, psi):
        """
        Loads a new set of parameters for the initial memory occupation of the FSC.

        Parameters:
        --- psi: np.array of shape (M)
            New parameters for the initial memory occupation.
        """
        self.psi = psi
        self.rho = fun.softmax(psi)
        
            
    def get_TMat(self, f):
        # check if f is a single feature vector or a sequence of feature vectors
        if len(f.shape) == 1:
            W = np.einsum('fmna, f->mna', self.theta, f)
        else:
            W = np.einsum('fmna, fz->zmna', self.theta, f)

        return fun.softmax(W, axis = (1, 2))
    

    def evaluate_nloglikelihood(self, trajectory):
        actions = trajectory["actions"]
        features = trajectory["features"]

        nLL = GenerationContinuousObs._nb_evaluate_nloglikelihood(GenerationContinuousObs._nb_get_TMat, features, actions, self.theta, self.rho,
                                               self.ActSpace, self.MemSpace, loaded_act_space = self.custom_act_space)

        return nLL



    @staticmethod
    @nb.njit
    def _nb_get_TMat(theta, f):
        M = theta.shape[1]
        A = theta.shape[3]
        F = theta.shape[0]

        W = np.zeros((M, M, A))

        for m in range(M):
            for n in range(M):
                for a in range(A):
                    for z in range(F):
                        W[m, n, a] += theta[z, m, n, a] * f[z]

        max_W = np.max(W)
        W = np.exp(W - max_W)

        for m in range(M):
            W[m] /= np.sum(W[m, :])

        return W


    @staticmethod
    @nb.njit
    def _nb_generate_trajectory(get_TMat_nb, NSteps, MSpace, MASpace, theta, rho, features):
        actions = np.zeros(NSteps, dtype = np.int32)
        memories = np.zeros(NSteps, dtype = np.int32)

        initial_memory = fun.numba_random_choice(MSpace, rho)

        memories[0] = initial_memory

        for t in range(0, NSteps):
            transition_probs = get_TMat_nb(theta, features[:, t])[memories[t]].flatten()
            new_MA = fun.numba_random_choice(MASpace, transition_probs)
            if t < NSteps - 1:
                memories[t + 1] = new_MA[0]
            actions[t] = new_MA[1]

        return actions, memories
    
    @staticmethod
    @nb.njit(parallel = True)
    def _nb_generate_trajectories_parallel(get_TMat_nb, NTraj, NSteps, MSpace, MASpace, theta, rho, features):
        actions = np.zeros((NTraj, NSteps), dtype = np.int32)
        memories = np.zeros((NTraj, NSteps), dtype = np.int32)

        for n in nb.prange(NTraj):
            initial_memory = fun.numba_random_choice(MSpace, rho)
    
            memories[n, 0] = initial_memory
            for t in range(0, NSteps):
                transition_probs = get_TMat_nb(theta, features[n][:, t])[memories[n, t]].flatten()
                new_MA = fun.numba_random_choice(MASpace, transition_probs)
                if t < NSteps - 1:
                    memories[n, t+1] = new_MA[0]
                actions[n, t] = new_MA[1]

        return actions, memories
    

    @staticmethod
    @nb.njit
    def _nb_evaluate_nloglikelihood(get_TMat_nb, features, actions, theta, rho, ActSpace, MemSpace,
                                    loaded_act_space = False):
            nLL = 0.
    
            for t in range(actions.size):
                a = actions[t]
                f = features[:, t]

                if loaded_act_space:
                    idx_a = np.where(ActSpace == a)[0][0]
                else:
                    idx_a = a

    
                transition_probs = get_TMat_nb(theta, f)[:, :, idx_a].T
                    
                if t == 0:
                    m = transition_probs @ rho
                else:
                    m = transition_probs @ m
    
                mv = np.sum(m)
                nLL = nLL - np.log(mv)
                m /= mv
    
            return nLL - np.log(np.sum(m))
    





class InferenceContinuousObs():

    def __init__(self, M, A, F,
                 ActSpace = None, MemSpace = None,
                 seed = None):
        """
        Creates an instance of a Finite State Controller (FSC) for discrete features. The parameters
        M, A, and Y are used to define the number of memory states, actions, and features, respectively.
        The FSC is then trained to learn the transition probabilities and the initial memory occupation from
        a set of trajectories, by minimizing the negative log-likelihood of the observed trajectories.

        The optimization is performed using PyTorch, and the model is trained using the Adam optimizer with
        a learning rate schedule.

        Parameters:
        --- M: int
            Number of memory states.
        --- A: int
            Number of actions.
        --- F: int
            Number of features.
        --- ActSpace: list (default = None)
            List of actions. If None, the actions are assumed to be integers from 0 to A-1.
        --- MemSpace: list (default = None)
            List of memory states. If None, the memory states are assumed to be integers from 0 to M-1.
        --- seed: int (default = None)
            Seed for the random number generator. If None, the seed is not set.
        """

        self.M = M
        self.A = A
        self.F = F

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
    
        if seed is not None:
            torch.manual_seed(seed)

        self.theta = nn.Parameter(torch.randn(self.F, self.M, self.M, self.A, device = self.device))
        self.psi = nn.Parameter(torch.randn(self.M, device = self.device))


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

        self.trajectories_loaded = False
        self.optimizer_initialized = False
        self.trained = False

    def get_TMat(self, f):
        if len(f.shape) == 1:
            W = torch.einsum('fmna, f -> mna', self.theta, f)
            TM = fun.torch_softmax_2dim(W, dims = (1, 2))
        else:
            W = torch.einsum('fmna, fz -> zmna', self.theta, f)
            TM = fun.torch_softmax_2dim(W, dims = (2, 3))

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
            f = torch.tensor(trajectory["features"]).to(self.device)
            act = torch.tensor(trajectory["actions"]).to(self.device)
            self.FeatAct_trajectories.append([f, act])
            
                    
        self.trajectories_loaded = True
    
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
        features, actions = self.FeatAct_trajectories[idx_traj]

        return self.loss(features, actions, grad_required = grad_required)
            
    def loss(self, features, actions, grad_required = True):
        nLL = torch.tensor(0.0, requires_grad = grad_required)

        TMat = self.get_TMat(features)

        for t in range(features.size(1)):
            idx_a = actions[t]

            transition_probs = TMat[t, :, :, idx_a].T

            if t == 0:
                m = torch.matmul(transition_probs, self.rho)
            else:
                m = torch.matmul(transition_probs, m)

            mv = torch.sum(m)
            nLL = nLL - torch.log(mv)
            m /= mv

        return nLL - torch.log(torch.sum(m))       

    def optimize(self, NEpochs, NBatch, lr, train_split = 0.8, optimizer = None, gamma = 0.9,
                 verbose = False):
        
        assert self.trajectories_loaded, "No trajectories have been loaded. Load trajectories with the load_trajectories method."
        assert self.trained == False, "The model has already been trained. If you want to train it again, reinitialize it or set the flag self.trained to False."

        if optimizer is not None:
            self.optimizer = optimizer([self.theta, self.psi], lr = lr)
        else:
            self.optimizer = torch.optim.Adam([self.theta, self.psi], lr = lr)

        # add an exponential scheduler
        scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = gamma)

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
                loss = torch.tensor(0.0, requires_grad = True)

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
    

    def load_theta(self, theta):
        """
        Loads a new set of parameters for the transition probabilities of the FSC.

        Parameters:
        --- theta: torch.tensor of shape (Y, M, M, A)
            New parameters for the transition probabilities.
        """
        self.theta = nn.Parameter(torch.tensor(theta, device = self.device))

    def load_psi(self, psi):
        """
        Loads a new set of parameters for the initial memory occupation of the FSC.

        Parameters:
        --- psi: torch.tensor of shape (M)
            New parameters for the initial memory occupation.
        """
        self.psi = nn.Parameter(torch.tensor(psi, device = self.device))
        self.rho = nn.Softmax(dim = 0)(self.psi)