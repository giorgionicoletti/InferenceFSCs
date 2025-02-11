import numpy as np
import torch
import utils

from DiscreteObs.generation import GenerationDiscreteObs
from DiscreteObs.inference import InferenceDiscreteObs

from ContinuousObs.generation import GenerationContinuousObs
from ContinuousObs.inference import InferenceContinuousObs
import warnings

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

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

    def get_TMat(self, f = None):
        if self.__obs_type == 'discrete':
            if f is not None:
                raise Warning("Features are not used in the transition model for discrete observations.")
            if self.mode == 'generation':
                return self.generator.get_TMat()
            elif self.mode == 'inference':
                return self.inferencer.get_TMat()
        elif self.__obs_type == 'continuous':
            if f is None:
                raise ValueError("Features must be provided to compute the transition model for continuous observations.")
            
            if self.mode == 'generation':
                return self.generator.get_TMat(f)
            elif self.mode == 'inference':
                return self.inferencer.get_TMat(f)

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
            self._fitted_features_numpy.append(features.detach().cpu().numpy().astype(np.float64))

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'generation':
            if self.__obs_type == 'discrete':
                # check if observations have been loaded
                if hasattr(self, '_fitted_observations_numpy'):
                    self.convert_observations_to_numpy()

                self.generator = GenerationDiscreteObs(self)
                if hasattr(self, '_fitted_observations_numpy'):
                    self.generator.load_observations(self._fitted_observations_numpy)
            else:
                #check if features have been loaded
                if hasattr(self, '_fitted_features_numpy'):
                    self.convert_features_to_numpy()
                # convert parameters to float64, if they are not already
                if self.theta is not None and self.psi is not None:
                    self.theta = self.theta.astype(np.float64)
                    self.psi = self.psi.astype(np.float64)

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

    def compute_loss(self, trajectories):
        nLL = 0.0

        if self.mode == 'inference':
            for trj in trajectories:
                features = trj["features"]
                actions = trj["actions"]

                nLL += self.inferencer.loss(features, actions, grad_required=False)
        else:
            for trj in trajectories:
                nLL += self.generator.evaluate_nloglikelihood(trj)

            nLL /= len(trajectories)

        return nLL

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
    
    def optimize_parameters_continuous(self, NEpochs, NBatch, lr, train_split=0.8, optimizer=None, scheduler = None, gamma=0.9, verbose=False,
                                       overwrite=True, use_penalty=False, pActEq_target=None, alpha=1.0):
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

        if use_penalty:
            pActEq_target = torch.tensor(pActEq_target, dtype=torch.float32)
            losses_train, losses_val = self.inferencer.optimize_with_penalty(NEpochs, NBatch, lr_theta, lr_psi, pActEq_target, alpha, train_split, optimizer, gamma, verbose)
        else:
            losses_train, losses_val = self.inferencer.optimize(NEpochs, NBatch, lr_theta, lr_psi, train_split, optimizer, scheduler, gamma, verbose)

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

    def fit(self, trajectories, NEpochs, NBatch, lr, train_split=0.8, optimizer=None, scheduler = None, gamma=0.9, verbose=False,
            maxiter=1000, rho0=None, th=1e-6, c_gauge=0, overwrite=True, use_penalty=False, pActEq_target=None, alpha=1.0):
        self.initialize_for_inference()
        
        self.load_trajectories_tofit(trajectories)
        
        self.__check_ready_for_inference()
        
        if self.__obs_type == 'discrete':
            if use_penalty:
                raise ValueError("Penalty is not yet supported for discrete observations.")
            if scheduler is not None:
                raise ValueError("Learning rate scheduler is not yet supported for discrete observations.")
            return self.optimize_parameters_discrete(NEpochs, NBatch, lr, train_split, optimizer, gamma, verbose, maxiter, rho0, th, c_gauge, overwrite)
        else:
            return self.optimize_parameters_continuous(NEpochs, NBatch, lr, train_split, optimizer, scheduler, gamma, verbose, overwrite, use_penalty, pActEq_target, alpha)


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

    def generate_trajectories(self, NSteps = None, observations=None, idx_observation=None, NTraj=None, verbose=False,
                              features=None, idx_feature=None):
        if self.mode != 'generation':
            raise ValueError("Mode must be 'generation' to generate trajectories.")
        if self.__obs_type == 'continuous':
            return self.generator.generate_trajectories(NSteps, features, idx_feature, NTraj, verbose)
        elif self.__obs_type == 'discrete':
            return self.generator.generate_trajectories(NSteps, observations, idx_observation, NTraj, verbose)

    def load_parameters(self, theta, psi):

        self.__check_parameters_consistency(theta, psi)

        if self.mode == 'inference':
            if isinstance(theta, np.ndarray):
                theta = torch.tensor(theta, dtype=torch.float32)
            if isinstance(psi, np.ndarray):
                psi = torch.tensor(psi, dtype=torch.float32)
        elif self.mode == 'generation':
            if isinstance(theta, torch.Tensor):
                theta = theta.detach().cpu().numpy().astype(np.float64)
            if isinstance(psi, torch.Tensor):
                psi = psi.detach().cpu().numpy().astype(np.float64)

        self.theta = theta
        self.psi = psi


    def plot_FSC(self, features = None, ax = None, figsize = None,
                 memory1_color='gray', memory2_color='gray',
                 action_r_color='lightblue', action_t_color='salmon',
                 title=""):
        if self.M != 2 and self.A != 2:
            raise ValueError("Plotting is only supported for FSCs with 2 memory states and 2 actions.")
        
        if features is None and self.__obs_type == 'continuous':
            raise ValueError("Features must be provided to plot the FSC for continuous observations.")
        
        TMat = self.get_TMat(features)
        pi_prob = TMat.sum(axis = 1)
        pitilde_prob = TMat / pi_prob[:, None, :]

        joint = pi_prob*np.ones(self.M)[..., None]/self.M
        pA = np.sum(joint, axis = 0)
        pM = np.sum(joint, axis = 1)

        log_arg = joint / (pA[None, :] * pM[:, None])
        MI_memact = np.sum(joint * np.log2(log_arg))
        
        if ax is None:
            if figsize is None:
                figsize = (5, 3)
            fig, ax = plt.subplots(1, 1, figsize = figsize)

        utils.plot_FSC_network(ax, pi_prob, pitilde_prob,
                               memory1_color, memory2_color,
                               action_r_color, action_t_color,
                               title)
        
        if ax is None:
            return fig, ax

    def plot_dashboard_M2A2(self, feature_array, trajectories,
                            run_exponent = None, bins_tumble_dur = None, h_tumble_dur = None,
                            nLL = None, title = "", return_results = False,
                            memory1_color='gray', memory2_color='gray',
                            action_r_color='lightblue', action_t_color='salmon'):
        if len(feature_array) != 2:
            raise ValueError("Dashboard only plots the topology of the FSC for 2 features.")

        if self.mode != 'generation':
            self.set_mode('generation')
            warnings.warn("Mode was set to 'generation' to plot the FSC.")
        
        
        if run_exponent == None:
            run_durations = np.concatenate([utils.filter_durations(utils.extract_durations(tr["actions"]), 0)[1:-1] for tr in trajectories])

            run_values, run_cumulative = utils.get_cumulative(run_durations)
            run_params, _ = curve_fit(utils.expcum_fit, run_values, run_cumulative, p0=[0.01])
            run_exponent = run_params[0]

        if h_tumble_dur is None or bins_tumble_dur is None:
            tumble_durations = np.concatenate([utils.filter_durations(utils.extract_durations(tr["actions"]), 1)[1:-1] for tr in trajectories])

            h_tumble_dur, bins_tumble_dur = np.histogram(tumble_durations, bins=np.arange(1, np.max(tumble_durations) + 1), density=True)
            bins_tumble_dur = bins_tumble_dur[:-1]

        if nLL is None:            
            nLL = self.compute_loss(trajectories)

        generated_tr = self.generate_trajectories(features = [tr["features"] for tr in trajectories])
        run_durations_gen = np.concatenate([utils.filter_durations(utils.extract_durations(tr["actions"]), 0)[1:-1] for tr in generated_tr])
        tumble_durations_gen = np.concatenate([utils.filter_durations(utils.extract_durations(tr["actions"]), 1)[1:-1] for tr in generated_tr])
        memory1_durations_gen = np.concatenate([utils.filter_durations(utils.extract_durations(tr["memories"]), 0)[1:-1] for tr in generated_tr])
        memory2_durations_gen = np.concatenate([utils.filter_durations(utils.extract_durations(tr["memories"]), 1)[1:-1] for tr in generated_tr])
        memory1_occupacy = np.mean([np.sum(tr["memories"] == 0)/len(tr["memories"]) for tr in generated_tr])
        memory2_occupacy = np.mean([np.sum(tr["memories"] == 1)/len(tr["memories"]) for tr in generated_tr])

        h_run_dur_gen, bins_run_dur_gen = np.histogram(run_durations_gen, bins=50, density=True)
        bins_run_dur_gen = (bins_run_dur_gen[1:] + bins_run_dur_gen[:-1]) / 2

        h_tumb_dur_gen, bins_tumb_dur_gen = np.histogram(tumble_durations_gen, bins=np.arange(1, np.max(tumble_durations_gen) + 1), density=True)
        bins_tumb_dur_gen = bins_tumb_dur_gen[:-1]

        h_mem1_dur_gen, bins_mem1_dur_gen = np.histogram(memory1_durations_gen, density=True)
        bins_mem1_dur_gen = (bins_mem1_dur_gen[1:] + bins_mem1_dur_gen[:-1]) / 2
        h_mem2_dur_gen, bins_mem2_dur_gen = np.histogram(memory2_durations_gen, density=True)
        bins_mem2_dur_gen = (bins_mem2_dur_gen[1:] + bins_mem2_dur_gen[:-1]) / 2




        fig, axs = plt.subplot_mosaic([["FSC1", "FSC1", "FSC2", "FSC2", "rho"],
                                       ["run_dist", "run_dist", "tumbl_dist", "memory_dist1", "memory_occ"],
                                       ["run_dist", "run_dist", "tumbl_dist", "memory_dist2", "memory_occ"]],
                                       figsize=(20, 7.5),
                                       gridspec_kw={'width_ratios': [1, 1, 1, 1, 1],
                                                    'height_ratios': [1.2, 0.5, 0.5]})
        
        for idx_y in range(feature_array.shape[1]):
            self.plot_FSC(features=feature_array[:, idx_y], ax=axs[f"FSC{idx_y+1}"],
                          title = f"FSC for $y = {np.round(feature_array[1][idx_y], 3)}$",
                          memory1_color=memory1_color, memory2_color=memory2_color,
                          action_r_color=action_r_color, action_t_color=action_t_color)
        if title == "":
            overall_title = f"Loss: {np.round(nLL, 4)}"
        else:
            overall_title = title + f" - Loss: {np.round(nLL, 4)}"
        fig.suptitle(overall_title, fontsize=12)


        axs["rho"].bar(np.arange(self.M), self.rho, color=[memory1_color, memory2_color], alpha = 0.5)
        axs["rho"].set_ylabel("Probability")
        axs["rho"].set_xticks(np.arange(self.M))
        axs["rho"].set_xticklabels([f"$M_{i+1}$" for i in range(self.M)])
        axs["rho"].set_yticks(np.arange(0, 1.1, 0.1))
        axs["rho"].title.set_text("Initial memory distribution")
        # add the value over each bar
        for i, val in enumerate(self.rho):
            axs["rho"].text(i, val + 0.01, f"{val:.2f}", ha='center')

        axs["run_dist"].bar(bins_run_dur_gen, h_run_dur_gen, width=np.diff(bins_run_dur_gen)[0], label="Generated",
                    color = 'lightblue', alpha = 0.8, lw = 1, edgecolor = 'w')
        axs["run_dist"].plot(bins_run_dur_gen, run_exponent * np.exp(-run_exponent * bins_run_dur_gen),
                                label=f"Data exponential fit, $\\tau = {np.round(1/run_exponent, 2)}$ fr", ls="--", color = 'black')
        axs["run_dist"].set_yscale('log')
        axs["run_dist"].set_xlabel("Run duration (frames)")
        axs["run_dist"].set_ylabel("Probability density")
        axs["run_dist"].legend()

        axs["tumbl_dist"].bar(bins_tumb_dur_gen, h_tumb_dur_gen, width=np.diff(bins_tumb_dur_gen)[0], label="Generated",
                        color = 'salmon', alpha = 0.8, lw = 1, edgecolor = 'w')
        axs["tumbl_dist"].scatter(bins_tumble_dur, h_tumble_dur, color = 'k', label="Data", s = 50)
        axs["tumbl_dist"].set_yscale('log')
        axs["tumbl_dist"].set_xlabel("Tumble duration (frames)")
        axs["tumbl_dist"].set_ylabel("Probability density")
        axs["tumbl_dist"].legend()

        axs["memory_dist1"].bar(bins_mem1_dur_gen, h_mem1_dur_gen, width=np.diff(bins_mem1_dur_gen)[0], label="Generated",
                        color = 'gray', alpha = 0.8, lw = 0.5, edgecolor = 'w')
        axs["memory_dist1"].set_yscale('log')
        axs["memory_dist1"].set_xlabel("$M_1$ duration (frames)")
        axs["memory_dist1"].set_ylabel("Probability density")

        axs["memory_dist2"].bar(bins_mem2_dur_gen, h_mem2_dur_gen, width=np.diff(bins_mem2_dur_gen)[0], label="Generated",
                                color = 'gray', alpha = 0.8, lw = 0.5, edgecolor = 'w')
        axs["memory_dist2"].set_yscale('log')
        axs["memory_dist2"].set_xlabel("$M_2$ duration (frames)")
        axs["memory_dist2"].set_ylabel("Probability density")

        axs["memory_occ"].bar(np.arange(self.M), [memory1_occupacy,  memory2_occupacy], color='gray', alpha = 0.5)
        axs["memory_occ"].set_ylabel("Memory occupacy ($p(M)$)")
        axs["memory_occ"].set_xticks(np.arange(self.M))
        axs["memory_occ"].set_xticklabels([f"$M_{i+1}$" for i in range(self.M)])
        axs["memory_occ"].set_yticks(np.arange(0, 1.1, 0.1))
        for i, val in enumerate([memory1_occupacy,  memory2_occupacy]):
            axs["memory_occ"].text(i, val + 0.01, f"{np.round(val, 2)}", ha='center')

        plt.subplots_adjust(wspace=0.5, hspace = 0.4)
        plt.show()

        if return_results:
            results = {"run_durations_gen": run_durations_gen, "tumble_durations_gen": tumble_durations_gen,
                       "memory1_durations_gen": memory1_durations_gen, "memory2_durations_gen": memory2_durations_gen,
                       "memory1_occupacy": memory1_occupacy, "memory2_occupacy": memory2_occupacy,
                       "nLL": nLL}
            return results


    def compute_eq_probability(self, feature_array):
        if self.__obs_type == 'discrete':
            raise ValueError("Equilibrium probability is not defined for discrete observations.")
        
        elif self.__obs_type == 'continuous':
            if self.mode == 'generation':
                return self.generator.compute_eq_probability(feature_array)
            elif self.mode == 'inference':
                return self.inferencer.compute_eq_probability(feature_array)