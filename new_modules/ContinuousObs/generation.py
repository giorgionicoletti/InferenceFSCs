import numpy as np
import numba as nb
import utils

import matplotlib.pyplot as plt

class GenerationContinuousObs:

    def __init__(self, FSC):
        """
        """
        self.FSC = FSC

        self.rho = utils.softmax(self.FSC.psi)

        self.InternalMemSpace = np.arange(self.FSC.M)
        self.InternalActSpace = np.arange(self.FSC.A)
        self.InternalMemActSpace = utils.combine_spaces(self.InternalMemSpace, self.InternalActSpace)

    def load_features(self, features):
        """
        Loads a sequence of features to be used to generate a trajectory.

        Parameters:
        --- features: list of np.arrays
            List of feature sequences. Each element of the list is an np.array of possibly different lengths,
            containing the features for a single trajectory.
        """
        for ftraj in features:
            if np.shape(ftraj)[0] != self.FSC.F:
                raise ValueError("Feature sequence must have the same number of features as the FSC.")

        self.features = features
        self.features_lengths = np.array([ftraj.shape[1] for ftraj in features])
        self.min_features_length = np.min(self.features_lengths)
        self.max_features_length = np.max(self.features_lengths)

    def generate_single_trajectory(self, NSteps, features=None, idx_feature=None):
        """
        Generates a single trajectory of NSteps length given a sequence of features. If no features are provided,
        the method uses the loaded features and generates a trajectory for the indexed feature sequence.

        The number of steps NSteps must be smaller or equal than the number of features in the provided feature
        sequence. Note that the trajectory is not stored, but returned as a dictionary.

        Parameters:
        --- NSteps: int
            Number of steps for the trajectory.
        --- features: np.array (default = None)
            Array of features. If None, the method uses the loaded features.
        --- idx_feature: int (default = None)
            Index of the feature sequence to use if no features are provided.

        Returns:
        --- trajectory: dict
            Dictionary containing the actions, memories, and features for the generated trajectory.
        """
        if features is None:
            assert hasattr(self, "features"), "No features have been loaded. Load features with the load_features method."
            assert idx_feature is not None, "If no features are provided, the idx_feature parameter must not be None."
            assert NSteps <= self.features[idx_feature].shape[1], "NSteps must be smaller or equal than the number of features."

            features_cut = self.features[idx_feature][:, :NSteps]
        else:
            assert NSteps <= features.shape[1], "NSteps must be smaller or equal than the number of features."
            features_cut = features[:, :NSteps]

        int_actions, int_memories = GenerationContinuousObs._nb_generate_trajectory(GenerationContinuousObs._nb_get_TMat,
                                                                                    NSteps, self.InternalMemSpace, self.InternalMemActSpace,
                                                                                    self.FSC.theta, self.rho, features_cut)
        actions = np.array([self.FSC.ActSpace[act] for act in int_actions])
        memories = np.array([self.FSC.MemSpace[mem] for mem in int_memories])
        
        trajectory = {"actions": actions, "memories": memories, "features": features_cut}

        return trajectory
    
    def generate_trajectories(self, NSteps, features=None, idx_feature=None, NTraj=None, verbose=False):
        """
        Generates NTraj trajectories of NSteps length given a sequence of features. If no features are provided,
        the method uses the loaded features. It is also possible to generate NTraj trajectories for the same feature
        sequence by providing the idx_feature parameter and setting the NTraj parameter.

        In any case, the number of steps NSteps must be smaller or equal than the number of features in the provided
        feature sequence. If NSteps is None, the function will use the length of each feature sequence.

        Note that the trajectories are not stored in the object, but returned as a list of dictionaries.

        Parameters:
        --- NSteps: int or None
            Number of steps for the trajectory. If None, the function uses the length of each feature sequence.
        --- features: list of np.arrays (default = None)
            List of feature sequences. If None, the method uses the loaded features.
        --- idx_feature: int (default = None)
            Index of the feature sequence to use if no features are provided.
        --- NTraj: int (default = None)
            Number of trajectories to generate. If None, the method generates one trajectory per feature sequence.
        --- verbose: bool (default = False)
            If True, prints information about the generation process.

        Returns:
        --- trajectories: list of dicts
            List of dictionaries containing the actions, memories, and features for each generated trajectory.
        """
        features_cut = []

        if NSteps is None:
            if features is None:
                if not hasattr(self, "features"):
                    raise ValueError("No features have been loaded or provided.")
                features = self.features
            feat_lengths = [feat.shape[1] for feat in features]
            features_cut = [feat[:, :length] for feat, length in zip(features, feat_lengths)]
            NTraj = len(features)
        else:
            if features is None:
                if idx_feature is None:
                    if verbose:
                        print("No features provided. Using the loaded features and generating one trajectory per feature sequence.")
                    assert hasattr(self, "features"), "No features have been loaded. Load features with the load_features method."
                    assert NSteps <= self.min_features_length, "NSteps must be smaller than the shortest feature length."
                    NTraj = len(self.features)
                    
                    for n in range(NTraj):
                        features_cut.append(self.features[n][:, :NSteps])

                else:
                    if verbose:
                        print("No features provided. Using the indexed feature sequence and generating NTraj trajectories for the same feature sequence.")
                    assert hasattr(self, "features"), "No features have been loaded. Load features with the load_features method."
                    assert NSteps <= self.features[idx_feature].shape[1], "NSteps must be smaller or equal than the number of features."
                    assert NTraj is not None, "If no features are provided, the NTraj parameter must be provided."

                    for n in range(NTraj):
                        features_cut.append(self.features[idx_feature][:, :NSteps])

            elif type(features[0]) is np.ndarray:
                if verbose:
                    print("Multiple feature sequences provided. Generating one trajectory per feature sequence.")
                feat_lengths = np.array([feat.shape[1] for feat in features])
                assert np.all(feat_lengths >= NSteps), "All feature sequences must have at least NSteps features."

                for n in range(len(features)):
                    features_cut.append(features[n][:, :NSteps])

                NTraj = len(features)

            else:
                if verbose:
                    print("Single feature sequence provided. Generating NTraj trajectories for the same feature sequence.")
                assert NSteps <= features.shape[1], "NSteps must be smaller or equal than the number of features."
                assert NTraj is not None, "If features is a single array, the NTraj parameter must be provided."

                for n in range(NTraj):
                    features_cut.append(features[:, :NSteps])

        if NSteps is not None:
            int_actions, int_memories = GenerationContinuousObs._nb_generate_trajectories_parallel(GenerationContinuousObs._nb_get_TMat,
                                                                                                NTraj, NSteps, self.InternalMemSpace, self.InternalMemActSpace,
                                                                                                self.FSC.theta, self.rho, features_cut)
        else:
            int_actions, int_memories = GenerationContinuousObs._nb_generate_trajectories_parallel_nosteps(GenerationContinuousObs._nb_get_TMat,
                                                                                                           feat_lengths, self.InternalMemSpace, self.InternalMemActSpace,
                                                                                                           self.FSC.theta, self.rho, features_cut)
        trajectories = []

        for n in range(NTraj):
            actions = np.array([self.FSC.ActSpace[act] for act in int_actions[n]])
            memories = np.array([self.FSC.MemSpace[mem] for mem in int_memories[n]])

            trajectory = {"actions": actions, "memories": memories, "features": features_cut[n]}
            trajectories.append(trajectory)
        
        return trajectories
    
    def evaluate_nloglikelihood(self, trajectory):
        """
        Evaluates the negative log-likelihood of a given trajectory.

        Parameters:
        --- trajectory: dict
            Dictionary containing the actions, memories, and features for the trajectory.

        Returns:
        --- nLL: float
            Negative log-likelihood of the trajectory.
        """
        actions = trajectory["actions"]
        features = trajectory["features"]

        actions = self.__map_act_to_internal_space(actions)

        nLL = GenerationContinuousObs._nb_evaluate_nloglikelihood(GenerationContinuousObs._nb_get_TMat, features, actions,
                                                                  self.FSC.theta, self.rho)

        return nLL
    
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

    def plot_trajectory(self, trj, Time=None, colors_features=None):
        """
        Plots the actions, memories, and features of a given trajectory.
        """
        if Time is None:
            Time = np.arange(trj["features"].shape[1])

        fig, ax = plt.subplots(3,1, figsize=(10,5))
        plt.subplots_adjust(hspace=0.5)

        if colors_features is None:
            colors_features = ['k' for i in range(self.FSC.F)]

        for f in range(self.FSC.F):
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
    
    def get_TMat(self, f):
        if len(f.shape) == 1:
            W = np.einsum('fmna, f->mna', self.FSC.theta, f)
            TA = utils.softmax(W, axis = (1, 2))
        else:
            W = np.einsum('fmna, fz->zmna',self.FSC.theta, f)
            TA = utils.softmax(W, axis = (2, 3))

        return TA
    
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
    def _nb_trajectory_step(get_TMat_nb, MSpace, MASpace, theta, m, f):
        transition_probs = get_TMat_nb(theta, f)[m].flatten()
        new_MA = utils.numba_random_choice(MASpace, transition_probs)

        return new_MA[0], new_MA[1]

    @staticmethod
    @nb.njit
    def _nb_generate_trajectory(get_TMat_nb, NSteps, MSpace, MASpace, theta, rho, features):
        actions = np.zeros(NSteps, dtype = np.int32)
        memories = np.zeros(NSteps, dtype = np.int32)

        initial_memory = utils.numba_random_choice(MSpace, rho)

        memories[0] = initial_memory

        for t in range(0, NSteps):
            transition_probs = get_TMat_nb(theta, features[:, t])[memories[t]].flatten()
            new_MA = utils.numba_random_choice(MASpace, transition_probs)
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
            initial_memory = utils.numba_random_choice(MSpace, rho)
    
            memories[n, 0] = initial_memory
            for t in range(0, NSteps):
                transition_probs = get_TMat_nb(theta, features[n][:, t])[memories[n, t]].flatten()
                new_MA = utils.numba_random_choice(MASpace, transition_probs)
                if t < NSteps - 1:
                    memories[n, t+1] = new_MA[0]
                actions[n, t] = new_MA[1]

        return actions, memories
    
    @staticmethod
    @nb.njit(parallel = True)
    def _nb_generate_trajectories_parallel_nosteps(get_TMat_nb, NSteps_list, MSpace, MASpace, theta, rho, features):
        actions = [np.zeros(NSteps, dtype = np.int32) for NSteps in NSteps_list]
        memories = [np.zeros(NSteps, dtype = np.int32) for NSteps in NSteps_list]

        NTraj = len(NSteps_list)

        for n in nb.prange(NTraj):
            initial_memory = utils.numba_random_choice(MSpace, rho)
    
            memories[n][0] = initial_memory
            for t in range(0, NSteps_list[n]):
                transition_probs = get_TMat_nb(theta, features[n][:, t])[memories[n][t]].flatten()
                new_MA = utils.numba_random_choice(MASpace, transition_probs)
                if t < NSteps_list[n] - 1:
                    memories[n][t+1] = new_MA[0]
                actions[n][t] = new_MA[1]

        return actions, memories

    

    @staticmethod
    @nb.njit
    def _nb_evaluate_nloglikelihood(get_TMat_nb, features, actions, theta, rho):
            nLL = 0.
    
            for t in range(actions.size):
                a = actions[t]
                f = features[:, t]
    
                transition_probs = get_TMat_nb(theta, f)[:, :, a].T
                    
                if t == 0:
                    m = transition_probs @ rho
                else:
                    m = transition_probs @ m
    
                mv = np.sum(m)
                nLL = nLL - np.log(mv)
                m /= mv
    
            return nLL - np.log(np.sum(m))
    

    def compute_eq_probability(self, feature_array, return_eigvals = False):
        TMat_all = self.get_TMat(feature_array)

        pActEq = np.zeros((feature_array.shape[1], self.FSC.A)) # this is the probability of action a given feature
        pMemEq = np.zeros((feature_array.shape[1], self.FSC.M)) # this is the probability of memory m given feature
        policy = np.zeros((feature_array.shape[1],self.FSC.M, self.FSC.A))

        if return_eigvals:
            Eig = np.zeros((feature_array.shape[1], self.FSC.M))

        for idx_y, y in enumerate(feature_array[1]):
            TMat = TMat_all[idx_y]
            qprob = np.sum(TMat, axis=-1)
            
            eigvals, eigvecs = np.linalg.eig(qprob.T)
            tol = 1e-5
            idx_eig = np.isclose(eigvals, 1, rtol=tol)
            while np.sum(idx_eig) > 1:
                tol *= 0.1
                idx_eig = np.isclose(eigvals, 1, rtol=tol)

            if return_eigvals:
                Eig[idx_y] = eigvals

            pMemEq[idx_y] = eigvecs[:, idx_eig].flatten()
            pMemEq[idx_y] /= np.sum(pMemEq[idx_y])

            policy[idx_y] = np.sum(TMat, axis=1)

            pActEq[idx_y] = np.matmul(policy[idx_y].T, pMemEq[idx_y])

        if return_eigvals:
            return pActEq, pMemEq, policy, Eig
        else:
            return pActEq, pMemEq, policy
    