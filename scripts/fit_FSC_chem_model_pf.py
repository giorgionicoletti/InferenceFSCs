import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../new_modules')

import ChemotaxisModels as chem
import FSC as controller

import time as measure_time

NTraj = 250

NSteps1 = 500
NSteps2 = 1500
NBurn = 1000

dt = 1e-2

tau_sub = 25
ttumble = 0.1

NRuns = 5
c0_pf_array = np.linspace(10, 200, NRuns)

np.random.seed(42)
c1_pf_array = np.random.uniform(0.5, 2, NRuns)

N_FSC = 20
seeds_FSC = np.arange(0, N_FSC)
F = 2
M = 2
A = 2

NEpochs = 15
NBatch = 25
lr = (0.05, 0.05)
gamma = 0.99
train_split = 0.8

for idx_run in range(NRuns):
    print(f"Run {idx_run}")
    c0_pf = c0_pf_array[idx_run]
    c1_pf = c1_pf_array[idx_run] * c0_pf
    #results_model = chem.gradx_ecoli3D(NRep = NTraj, NSteps = NSteps, grad = grad, ttumble = ttumble)

    results_model_pf = chem.cswitch_ecoli3D(NRep = NTraj, NSteps1 = NSteps1, NSteps2 = NSteps2,
                                            c0 = c0_pf, c1 = c1_pf, ttumble = ttumble, NBurn = NBurn, dt = dt)

    c_data = [res["concentrations"][::tau_sub] for res in results_model_pf]
    cmean = np.concatenate(c_data).mean()

    c_data = [x/c0_pf for x in c_data]

    trajectories_data = []
    actions = []
    for i, curr_res in enumerate(results_model_pf):

        actions.append(curr_res["actions"].astype(int)[::tau_sub])
        ccurr = c_data[i]

        dict_traj = {}
        dict_traj["actions"] = actions[-1]
        dict_traj["features"] = np.array([np.ones(ccurr.size).astype(np.float32),
                                        ccurr])

        trajectories_data.append(dict_traj)

    first_action = np.array([tr["actions"][0] for tr in trajectories_data])

    print("Number of trajectories: ", len(trajectories_data))
    print("Number of tumbling at the beginning: ", np.sum(first_action))
    print("Fraction of tumbling at the beginning: ", np.round(np.sum(first_action) / len(trajectories_data) * 100, 2), "%")

    # save the data
    np.savez(f"../data/model/trajectories_model_pf_{idx_run}_c0{c0_pf}_c1{c1_pf}_ttumble{ttumble}_dt{dt}_tau_sub{tau_sub}.npz",
             trajectories_data = trajectories_data, c0_pf = c0_pf, c1_pf = c1_pf, ttumble = ttumble, dt = dt,
             NSteps1 = NSteps1, NSteps2 = NSteps2, NBurn = NBurn, tau_sub = tau_sub)

    for seed in seeds_FSC:
        tic = measure_time.time()
        FSC_tofit = controller.FSC("continuous", M = M, A = A, F = F, seed = seed)

        tloss, vloss = FSC_tofit.fit(trajectories_data, NEpochs = NEpochs,
                                    NBatch = NBatch, lr = lr, gamma = gamma, train_split = train_split)

        par_names = f"../data/parameters/FSC_M{M}_A{A}_F{F}_model_pf_run{idx_run}_c0{c0_pf}_c1{c1_pf}_ttumble{ttumble}_dt{dt}_tau_sub{tau_sub}_"
        par_names += f"seed_{seed}_NEpochs{NEpochs}_NTrajs{len(trajectories_data)}_"

        parameters = FSC_tofit.get_learned_parameters()

        # parameters is a dictionary with keys "theta" and "psi", add
        # the seed to the dictionary, the tloss and vloss
        parameters["seed"] = seed
        parameters["tloss"] = tloss
        parameters["vloss"] = vloss
        parameters["NEpochs"] = NEpochs
        parameters["NBatch"] = NBatch
        parameters["lr"] = lr
        parameters["gamma"] = gamma
        parameters["train_split"] = train_split

        # use npz format to save the parameters
        np.savez(par_names + "parameters.npz", **parameters)
        toc = measure_time.time()

        print(f"Seed {seed} done in {toc - tic} seconds")
        print()