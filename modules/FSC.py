import numpy as np
import numba as nb
import fun

class FSC_DiscreteObs():

    def __init__(self, theta, psi, verbose = False):
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
        self.A = theta.shape[0]
        self.Y = theta.shape[3]

        if verbose:
            print(f"Initializing FSC with {self.M} memory states, {self.A} actions, and {self.Y} observations.")

        assert self.M == theta.shape[1]
        assert self.M == theta.shape[2]

        self.ObsSpace = np.arange(self.Y)
        self.ActSpace = np.arange(self.A)
        self.MemSpace = np.arange(self.M)

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

    def generate_trajectory(self, NSteps):
        """
        Generates a trajectory of length NSteps using the FSC.

        Parameters:
        --- NSteps: int
            Length of the trajectory to be generated.

        Returns:
        --- trajectory: dict
            Dictionary containing the generated trajectory, with the following
            keys:
            --- "actions": np.array of shape (NSteps)
                Sequence of actions taken.
            --- "memories": np.array of shape (NSteps)
                Sequence of memory states.
            --- "observations": np.array of shape (NSteps)
                Sequence of observations.
        """
        # check that observations have been loaded, otherwise raise an error with a message
        assert hasattr(self, "observations"), "No observations have been loaded. Load observations with the load_observations method."
        assert NSteps < self.observations.size, "NSteps must be smaller than the number of observations."

        MASpace = fun.combine_spaces(self.MemSpace, self.ActSpace)
        observations_cut = self.observations[:NSteps]

        actions, memories = self.nb_generate_trajectory(NSteps, self.MemSpace, MASpace, self.TMat, self.rho, observations_cut)
        trajectory = {"actions": actions, "memories": memories, "observations": observations_cut}

        return trajectory

    @staticmethod
    @nb.njit
    def nb_generate_trajectory(NSteps, MSpace, MASpace, TMat, rho, observations):

        actions = np.zeros(NSteps, dtype = np.int32)
        memories = np.zeros(NSteps, dtype = np.int32)

        initial_memory = fun.numba_random_choice(MSpace, rho)

        memories[0] = initial_memory
        actions[0] = -1

        for t in range(1, NSteps):
            transition_probs = TMat[observations[t], memories[t-1]].flatten()
            new_MA = fun.numba_random_choice(MASpace, transition_probs)
            memories[t] = new_MA[0]
            actions[t] = new_MA[1]

        return actions, memories


    
