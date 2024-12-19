import numpy as np
import numba as nb
import utils

import matplotlib.pyplot as plt

class GenerationDiscreteObs:

    def __init__(self, FSC):
        """
        """
        self.FSC = FSC

        self.rho = utils.softmax(self.FSC.psi)
        self.TMat = utils.softmax(self.FSC.theta, axis = (2, 3))

        self.InternalMemSpace = np.arange(self.FSC.M)
        self.InternalActSpace = np.arange(self.FSC.A)
        self.InternalObsSpace = np.arange(self.FSC.Y)
        self.InternalMemActSpace = utils.combine_spaces(self.InternalMemSpace, self.InternalActSpace)

    def load_observations(self, observations):
        """
        Loads a sequence of observations to be used to generate a trajectory.

        Parameters:
        --- observations: list of np.arrays
            List of observation sequences. Each element of the list is an np.array of possibly different lengths,
            containing the observations for a single trajectory.
        """

        self.observations = []

        if type(observations[0]) is np.ndarray or type(observations[0]) is list:
            for obs_seq in observations:
                for obs in obs_seq:
                    assert obs in self.FSC.ObsSpace, "All observations must be in the observation space."
                self.observations.append(self.__map_obs_to_internal_space(np.array(obs_seq)))
        else:
            for obs in observations:
                assert obs in self.FSC.ObsSpace, "All observations must be in the observation space."

            self.observations.append(self.__map_obs_to_internal_space(np.array(observations)))

        self.observations_lengths = np.array([len(obs) for obs in observations])
        self.min_obs_length = np.min(self.observations_lengths)
        self.max_obs_length = np.max(self.observations_lengths)

    def generate_single_trajectory(self, NSteps, observations = None, idx_observation = None):
        """
        Generates a single trajectory of NSteps length given a sequence of observations. If no observations are provided,
        the method uses the loaded observations and generates a trajectory for the indexed observation sequence.

        The number of steps NSteps must be smaller or equal than the number of observations in the provided observation
        sequence. Note that the trajectory is not stored, but returned as a dictionary.

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

        int_actions, int_memories = GenerationDiscreteObs._nb_generate_trajectory(NSteps, self.InternalMemSpace, self.InternalMemActSpace,
                                                                                  self.TMat, self.rho, observations_cut)
        actions = np.array([self.FSC.ActSpace[act] for act in int_actions])
        memories = np.array([self.FSC.MemSpace[mem] for mem in int_memories])
        obs = np.array([self.FSC.ObsSpace[obs] for obs in observations_cut])
        
        trajectory = {"actions": actions, "memories": memories, "observations": obs}

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
        observations_cut = []

        if observations is None:
            if idx_observation is None:
                if verbose:
                    print("No observations provided. Using the loaded observations and generating one trajectory per observation sequence.")
                assert hasattr(self, "observations"), "No observations have been loaded. Load observations with the load_observations method."
                assert NSteps <= self.min_obs_length, "NSteps must be smaller than the shortest observation length."
                NTraj = len(self.observations)
                
                for n in range(NTraj):
                    observations_cut.append(self.observations[n][:NSteps])

            else:
                if verbose:
                    print("No observations provided. Using the indexed observation sequence and generating NTraj trajectories for the same observation sequence.")
                assert hasattr(self, "observations"), "No observations have been loaded. Load observations with the load_observations method."
                assert NSteps <= self.observations[idx_observation].size, "NSteps must be smaller or equal than the number of observations."
                assert NTraj is not None, "If no observations are provided, the NTraj parameter must be provided."

                for n in range(NTraj):
                    observations_cut.append(self.observations[idx_observation][:NSteps])

        elif type(observations[0]) is np.ndarray:
            if verbose:
                print("Multiple observation sequences provided. Generating one trajectory per observation sequence.")
            obs_lengths = np.array([len(obs) for obs in observations])
            assert np.all(obs_lengths >= NSteps), "All observation sequences must have at least NSteps observations."

            for n in range(len(observations)):
                observations_cut.append(observations[n][:NSteps])

            NTraj = len(observations)

        else:
            if verbose:
                print("Single observation sequence provided. Generating NTraj trajectories for the same observation sequence.")
            assert NSteps <= observations.size, "NSteps must be smaller or equal than the number of observations."
            assert NTraj is not None, "If observations is a single array, the NTraj parameter must be provided."

            for n in range(NTraj):
                observations_cut.append(observations[:NSteps])


        int_actions, int_memories = GenerationDiscreteObs._nb_generate_trajectories_parallel(NTraj, NSteps, self.InternalActSpace, self.InternalMemActSpace,
                                                                                             self.TMat, self.rho, observations_cut)
        trajectories = []

        for n in range(NTraj):
            actions = np.array([self.FSC.ActSpace[act] for act in int_actions[n]])
            memories = np.array([self.FSC.MemSpace[mem] for mem in int_memories[n]])
            obs = np.array([self.FSC.ObsSpace[obs] for obs in observations_cut[n]])

            trajectory = {"actions": actions, "memories": memories, "observations": obs}
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

        actions = self.__map_act_to_internal_space(actions)
        observations = self.__map_obs_to_internal_space(observations)

        nLL = GenerationDiscreteObs._nb_evaluate_nloglikelihood(observations, actions, self.TMat, self.rho)

        return nLL
    
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

    def plot_trajectory(self, trj, Time = None):
        """
        Plots the actions, memories, and observations of a given trajectory.
        """

        if Time is None:
            Time = np.arange(len(trj["observations"]))

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
    
    # def load_theta(self, theta):
    #     """
    #     Loads a new set of parameters for the transition probabilities of the FSC.

    #     Parameters:
    #     --- theta: np.array of shape (Y, M, M, A)
    #         New parameters for the transition probabilities.
    #     """
    #     self.theta = theta
    #     self.TMat = fun.softmax(theta, axis = (2, 3))
    #     self.policy = np.sum(self.TMat, axis = 2)

    # def load_psi(self, psi):
    #     """
    #     Loads a new set of parameters for the initial memory occupation of the FSC.

    #     Parameters:
    #     --- psi: np.array of shape (M)
    #         New parameters for the initial memory occupation.
    #     """
    #     self.psi = psi
    #     self.rho = fun.softmax(psi)
    

    @staticmethod
    @nb.njit
    def _nb_evaluate_nloglikelihood(observations, actions, TMat, rho):
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

            Returns:
            --- nLL: float
                Negative log-likelihood of the trajectory.
            """
            nLL = 0.
    
            for t, obs in enumerate(observations):
                a = actions[t]
    
                transition_probs = TMat[obs, :, :, a].T
    
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

        initial_memory = utils.numba_random_choice(MSpace, rho)

        memories[0] = initial_memory

        for t in range(0, NSteps):
            transition_probs = TMat[observations[t], memories[t]].flatten()
            new_MA = utils.numba_random_choice(MASpace, transition_probs)
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
                initial_memory = utils.numba_random_choice(MSpace, rho)
    
                memories[n, 0] = initial_memory
    
                for t in range(0, NSteps):
                    transition_probs = TMat[observations[n][t], memories[n][t]].flatten()
                    new_MA = utils.numba_random_choice(MASpace, transition_probs)
                    if t < NSteps - 1:
                        memories[n, t+1] = new_MA[0]
                    actions[n, t] = new_MA[1]
    
            return actions, memories