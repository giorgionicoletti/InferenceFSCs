import numpy as np
import torch
import utils

from DiscreteObs.generation import GenerationDiscreteObs
from DiscreteObs.inference import InferenceDiscreteObs

from ContinuousObs.generation import GenerationContinuousObs
from ContinuousObs.inference import InferenceContinuousObs

class FSC:
    def __init__(self, obs_type, M, A, Y = None, F = None,
                 mode = 'inference',
                 theta = None, psi = None, seed = None,
                 ObsSpace = None, ActSpace = None, MemSpace = None,
                 verbose = False):
        
        self.__obs_type = obs_type
        self.__check_obs_consistency(Y, F)
        self.__initialize_structure(M, A, Y, F)

        self.__check_parameters_consistency(theta, psi)

        if theta is not None and psi is not None:
            self.theta = theta
            self.psi = psi
        elif theta is None and psi is not None:
            self.psi = psi
            self.__initialize_parameters(seed, which='theta')
        elif theta is not None and psi is None:
            self.theta = theta
            self.__initialize_parameters(seed, which='psi')
        else:
            self.__initialize_parameters(seed, which='both')

        self.__initialize_spaces(ObsSpace, ActSpace, MemSpace)

        self.mode = mode

        if mode == 'generation':
            if self.__obs_type == 'discrete':
                self.generator = GenerationDiscreteObs(self)
            else:
                self.generator = GenerationContinuousObs(self)
        elif mode == 'inference':
            if self.__obs_type == 'discrete':
                self.inferencer = InferenceDiscreteObs(self)
            else:
                self.inferencer = InferenceContinuousObs(self)
        else:
            raise ValueError("Mode must be either 'generation' or 'inference'")
        
        self.__loaded_trajectories_inference = False

    @property
    def obs_type(self):
        return self.__obs_type
    
    @property
    def rho(self):
        return utils.softmax(self.psi)

    @property
    def TMats(self, f = None):
        if self.__obs_type == 'discrete':
            if f is None:
                raise Warning("Features are not used in the transition model for discrete observations.")
            return utils.softmax(self.theta, axis = (2, 3))
        else:
            if f is None:
                raise ValueError("Features must be provided to compute the transition model for continuous observations.")
            
            W = np.einsum('fmna, f->mna', self.theta, f)
            return utils.softmax(W, axis = (1, 2))

    def __initialize_structure(self, M, A, Y, F):
        self.M = M
        self.A = A
        if self.__obs_type == 'discrete':
            self.Y = Y
        elif self.__obs_type == 'continuous':
            self.F = F

    def __initialize_parameters(self, seed, which = 'both'):
        if seed is not None:
            np.random.seed(seed)

        if which == 'both' or which == 'theta':
            if self.__obs_type == 'discrete':
                self.theta = np.random.randn(self.Y, self.M, self.M, self.A)
            elif self.__obs_type == 'continuous':
                self.theta = np.random.randn(self.F, self.M, self.M, self.A)

        if which == 'both' or which == 'psi':
            self.psi = np.random.randn(self.M)

    def __check_obs_consistency(self, Y, F):
        if self.__obs_type not in ['discrete', 'continuous']:
            raise ValueError("obs_type must be either 'discrete' or 'continuous'.")
        if Y is not None and self.__obs_type != 'discrete':
            raise ValueError("Y is provided but obs_type is not 'discrete'. Discrete observations are required for setting the number of possible observations Y.")
        if Y is None and self.__obs_type == 'discrete':
            raise ValueError("Y is not provided but obs_type is 'discrete'. Please provide the number of possible observations Y.")
        if F is not None and self.__obs_type != 'continuous':
            raise ValueError("F is provided but obs_type is not 'continuous'. Continuous observations are required for setting the number of features F.")
        if F is None and self.__obs_type == 'continuous':
            raise ValueError("F is not provided but obs_type is 'continuous'. Please provide the number of features F.")
        if Y is None and F is None:
            raise ValueError("Either the number of possible observations Y or the number of features F must be provided.")

    def __check_parameters_consistency(self, theta, psi):
        if self.__obs_type == 'discrete':
            if theta is not None and theta.shape != (self.Y, self.M, self.M, self.A):
                raise ValueError("theta must have shape (Y, M, M, A) where Y is the number of possible observations, M is the number of states, M is the number of actions, and A is the number of actions.")
        elif self.__obs_type == 'continuous':
            if theta is not None and theta.shape != (self.F, self.M, self.M, self.A):
                raise ValueError("theta must have shape (F, M, M, A) where F is the number of features, M is the number of states, M is the number of actions, and A is the number of actions.")
        if psi is not None and psi.shape != (self.M,):
            raise ValueError("psi must have shape (M,) where M is the number of states.")
        
    def __initialize_spaces(self, ObsSpace, ActSpace, MemSpace):
        if self.__obs_type == 'discrete':
            if ObsSpace is not None:
                if len(ObsSpace) != self.Y:
                    raise ValueError("The number of observations in ObsSpace must match the number of observations.")
                self.ObsSpace = np.array(ObsSpace)
                self.custom_obs_space = True
            else:
                self.ObsSpace = np.arange(self.Y)
                self.custom_obs_space = False

        if ActSpace is not None:
            if len(ActSpace) != self.A:
                raise ValueError("The number of actions in ActSpace must match the number of actions.")
            self.ActSpace = np.array(ActSpace)
            self.custom_act_space = True
        else:
            self.ActSpace = np.arange(self.A)
            self.custom_act_space = False

        if MemSpace is not None:
            if len(MemSpace) != self.M:
                raise ValueError("The number of states in MemSpace must match the number of states.")
            self.MemSpace = np.array(MemSpace)
            self.custom_mem_space = True
        else:
            self.MemSpace = np.arange(self.M)
            self.custom_mem_space = False
    
    def convert_observations_to_numpy(self):
        if not self.__loaded_trajectories_inference:
            raise ValueError("No trajectories to fit have been loaded.")
        
        self._fitted_observations_numpy = []
        for obs, _ in self.inferencer.ObsAct_trajectories:
            obs_original_space = np.array([self.ObsSpace[o.item()] for o in obs])
            self._fitted_observations_numpy.append(obs_original_space)

    def convert_features_to_numpy(self):
        if not self.__loaded_trajectories_inference:
            raise ValueError("No trajectories to fit have been loaded.")
        
        self._fitted_features_numpy = []
        for features, _ in self.inferencer.FeatAct_trajectories:
            self._fitted_features_numpy.append(features.detach().cpu().numpy())

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'generation':
            if self.__obs_type == 'discrete':
                self.convert_observations_to_numpy()

                self.generator = GenerationDiscreteObs(self)
                if hasattr(self, '_fitted_observations_numpy'):
                    self.generator.load_observations(self._fitted_observations_numpy)
            else:
                self.convert_features_to_numpy()

                self.generator = GenerationContinuousObs(self)
                if hasattr(self, '_fitted_features_numpy'):
                    self.generator.load_features(self._fitted_features_numpy)
        elif mode == 'inference':
            if self.__obs_type == 'discrete':
                self.inferencer = InferenceDiscreteObs(self)

            else:
                self.inferencer = InferenceContinuousObs(self)


    def load_trajectories_tofit(self, trajectories):
        if self.mode != 'inference':
            raise ValueError("Mode must be 'inference' to load trajectories to be fitted.")
        self.inferencer.load_trajectories(trajectories)
        self.__loaded_trajectories_inference = True

    @property
    def trajectories_tofit(self):
        if not self.__loaded_trajectories_inference:
            raise ValueError("No trajectories to fit have been loaded.")

        if self.__obs_type == 'discrete':
            return self.inferencer.ObsAct_trajectories
        else:
            return self.inferencer.FeatAct_trajectories

    def compute_loss(self):
        nLL = 0.0

        if self.mode == 'inference':
            if self.__obs_type == 'discrete':
                for idx in range(len(self.inferencer.ObsAct_trajectories)):
                    nLL += self.inferencer.evaluate_nloglikelihood(idx)

                nLL /= len(self.inferencer.ObsAct_trajectories)
            else:
                for idx in range(len(self.inferencer.FeatAct_trajectories)):
                    nLL += self.inferencer.evaluate_nloglikelihood(idx)

                nLL /= len(self.inferencer.FeatAct_trajectories)

        else:
            raise ValueError("Loss cannot be computed in generation mode. Set mode to 'inference' to compute the loss.")
        
        return

    def optimize_parameters_discrete(self, NEpochs, NBatch, lr, train_split=0.8, optimizer=None, gamma=0.9, verbose=False,
                                     maxiter=1000, rho0=None, th=1e-6, c_gauge=0, overwrite=True):
        if self.mode != 'inference':
            raise ValueError("Mode must be 'inference' to optimize parameters.")
        if not self.__loaded_trajectories_inference:
            raise ValueError("No trajectories to fit have been loaded.")

        losses_train, losses_val = self.inferencer.optimize(NEpochs, NBatch, lr, train_split, optimizer, gamma, verbose,
                                                            maxiter, rho0, th, c_gauge)

        if overwrite:
            self.theta = self.inferencer.theta.detach().cpu().numpy()
            self.psi = self.inferencer.psi.detach().cpu().numpy()

        return losses_train, losses_val
    
    def optimize_parameters_continuous(self, NEpochs, NBatch, lr, train_split=0.8, optimizer=None, gamma=0.9, verbose=False,
                                       overwrite=True):
        if self.mode != 'inference':
            raise ValueError("Mode must be 'inference' to optimize parameters.")
        if not self.__loaded_trajectories_inference:
            raise ValueError("No trajectories to fit have been loaded.")
        
        # check if lr is a number or a tuple
        if isinstance(lr, tuple):
            lr_theta, lr_psi = lr
        else:
            lr_theta = lr
            lr_psi = lr
        
        losses_train, losses_val = self.inferencer.optimize(NEpochs, NBatch, lr_theta, lr_psi, train_split, optimizer, gamma, verbose)

        if overwrite:
            self.theta = self.inferencer.theta.detach().cpu().numpy()
            self.psi = self.inferencer.psi.detach().cpu().numpy()

        return losses_train, losses_val

    def evaluate_nloglikelihood_for_trajectory(self, trajectory_idx):
        if self.mode != 'inference':
            raise ValueError("Mode must be 'inference' to evaluate the negative log-likelihood.")
        if not self.__loaded_trajectories_inference:
            raise ValueError("No trajectories to fit have been loaded.")
        return self.inferencer.evaluate_nloglikelihood(trajectory_idx)

    def get_learned_parameters(self):
        if self.mode != 'inference':
            raise ValueError("Mode must be 'inference' to retrieve learned parameters.")
        return {
            "theta": self.inferencer.theta.detach().cpu().numpy(),
            "psi": self.inferencer.psi.detach().cpu().numpy()
        }

    def initialize_for_inference(self):
        self.set_mode('inference')
        self.__loaded_trajectories_inference = False

    def fit(self, trajectories, NEpochs, NBatch, lr, train_split=0.8, optimizer=None, gamma=0.9, verbose=False,
            maxiter=1000, rho0=None, th=1e-6, c_gauge=0, overwrite=True):
        self.initialize_for_inference()
        
        self.load_trajectories_tofit(trajectories)
        
        self.__check_ready_for_inference()
        
        if self.__obs_type == 'discrete':
            return self.optimize_parameters_discrete(NEpochs, NBatch, lr, train_split, optimizer, gamma, verbose, maxiter, rho0, th, c_gauge, overwrite)
        else:
            return self.optimize_parameters_continuous(NEpochs, NBatch, lr, train_split, optimizer, gamma, verbose, overwrite)


    def __check_ready_for_inference(self):
        if self.mode != 'inference':
            raise ValueError("Mode must be 'inference' to perform inference.")
        if not self.__loaded_trajectories_inference:
            raise ValueError("No trajectories to fit have been loaded.")
        if not hasattr(self, 'inferencer'):
            raise ValueError("Inferencer has not been initialized.")

    def load_observations(self, observations):
        if self.mode != 'generation':
            raise ValueError("Mode must be 'generation' to load observations.")
        if self.__obs_type == 'continuous':
            raise ValueError("Observations can only be loaded for discrete observation. Use load_features instead.")

        self.generator.load_observations(observations)

    def load_features(self, features):
        if self.mode != 'generation':
            raise ValueError("Mode must be 'generation' to load features.")
        if self.__obs_type == 'discrete':
            raise ValueError("Features can only be loaded for continuous observation. Use load_observations instead.")

        self.generator.load_features(features)

    def generate_single_trajectory(self, NSteps, observations=None, idx_observation=None):
        if self.mode != 'generation':
            raise ValueError("Mode must be 'generation' to generate a trajectory.")
        return self.generator.generate_single_trajectory(NSteps, observations, idx_observation)

    def generate_trajectories(self, NSteps, observations=None, idx_observation=None, NTraj=None, verbose=False):
        if self.mode != 'generation':
            raise ValueError("Mode must be 'generation' to generate trajectories.")
        return self.generator.generate_trajectories(NSteps, observations, idx_observation, NTraj, verbose)

    def load_parameters(self, theta, psi):

        self.__check_parameters_consistency(theta, psi)

        if self.mode == 'inference':
            self.mode = 'generation'
            raise Warning("Mode has been changed to 'generation' to load parameters. Please set mode to 'inference' to optimize parameters.")
        
        if isinstance(theta, torch.Tensor):
            theta = theta.detach().cpu().numpy()
        if isinstance(psi, torch.Tensor):
            psi = psi.detach().cpu().numpy()

        self.theta = theta
        self.psi = psi