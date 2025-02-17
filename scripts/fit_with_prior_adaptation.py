import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../new_modules')
import ObsModels
import utils

import ChemotaxisModels as chem

import FSC as controller

from scipy.optimize import curve_fit
import numba as nb

import time as measure_time
import pickle

NRep = 1000
NSteps1 = 10000
NSteps2 = 10000
c0 = 0.2
c1 = c0*10
ttumble = 0.1
seeds = np.arange(NRep)
dt = 1e-3

tau_sub = 100

F = 2
M = 2
A = 2

cvalues = np.array([0.1, 1])
feature_array = np.array([np.ones(cvalues.size), cvalues])

psi_design = np.array([3, 1])

theta_design = np.array([[[[0.5, 0.5], [0.5, 0.5]],
                          [[0.5, 0.5], [0.5, 0.5]]],
                          [[[0.5, 0.5], [0.5, 0.5]],
                          [[0.5, 0.5], [0.5, 0.5]]]])
theta_design[0, 0, 0, 0] = 2.7
theta_design[0, 0, 1, 0] = -3.5
theta_design[0, 0, 1, 1] = -2
theta_design[0, 1, 0, 1] = -2

theta_design[0, 1, 1, 0] = 2.5
theta_design[0, 1, 0, 0] = 0.005

theta_design[1, 0, 0, 0] = 5
theta_design[1, 0, 1, 0] = 8.3 ###

theta_design[1, 1, 1, 0] = 0.87
theta_design[1, 1, 0, 0] = -2

theta_design[1, 1, 0, 1] = -8


FSC_designed = controller.FSC("continuous", M = M, A = A, F = F, mode = "generation")
FSC_designed.load_parameters(theta_design, psi_design)

Ntraj_play = 500
concentration_low = np.ones(NSteps1//tau_sub) * 0.1
concentration_high = np.ones(NSteps2//tau_sub) * 1
concentration = np.concatenate((concentration_low, concentration_high))

features_play = [np.array([np.ones(concentration.size), concentration])]*Ntraj_play

generated_tr = FSC_designed.generate_trajectories(features = features_play)

for i in range(len(generated_tr)):
    kkeys = list(generated_tr[i].keys())
    generated_tr[i]["actions"] = generated_tr[i]["actions"][NSteps1//2//tau_sub:]
    generated_tr[i]["memories"] = generated_tr[i]["memories"][NSteps1//2//tau_sub:]
    generated_tr[i]["features"] = generated_tr[i]["features"][:, NSteps1//2//tau_sub:]


seeds_list = np.arange(100)
biases_list = [-6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5]

save_path = "../data/prior_bias/"
save_path += f"FSC_params_prior_bias_adaptation_M{M}_A{A}_F{F}_"

for bias in biases_list:
    FSC_inferred_list = []

    for seed in seeds_list:
        print(f"Training - Seed {seed}, bias {bias}")
        np.random.seed(seed)

        theta_prior = np.random.randn(F, M, M, A)

        theta_prior[:, 0, 1, :] = np.random.randn(F, A) + bias
        theta_prior[:, 1, 0, :] = np.random.randn(F, A) + bias

        psi_prior = np.random.randn(M)

        FSC_prior = controller.FSC("continuous", M = M, A = A, F = F, mode = "generation")
        FSC_prior.load_parameters(theta_prior, psi_prior)
        
        tic = measure_time.time()
        tloss, vloss = FSC_prior.fit(generated_tr,
                                     NEpochs = 20,
                                     NBatch = 25, lr = (0.05, 0.05),
                                     scheduler = "exp",
                                     gamma = 0.99, train_split = 0.9)
        learned_params = FSC_prior.get_learned_parameters()
        toc = measure_time.time()

        print(f"Seed {seed} - Training time: {toc - tic:.2f} s")
        
        FSC_inferred_list.append(learned_params)

        print("\n\n")

    with open(save_path + f"bias_{bias}.pkl", "wb") as f:
        pickle.dump(FSC_inferred_list, f)