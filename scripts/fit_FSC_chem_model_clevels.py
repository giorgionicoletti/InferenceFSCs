import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../new_modules')

import ChemotaxisModels as chem
import FSC as controller

import time as measure_time
import os

NTraj = 200
NSteps = 8000
NBurn = 10000

NSteps1 = NSteps - 100
NSteps2 = 100

dt = 1e-3

tau_sub = 100
ttumble = 0.1

NLevels = 2
cmin = 10
cmax = 400
cvalues = np.linspace(cmin, cmax, NLevels)

c_array = np.random.choice(cvalues, NTraj)

results_model_pf = chem.cswitch_ecoli3D(NRep = NTraj, NSteps1 = NSteps1, NSteps2 = NSteps2,
                                        c0 = c_array, c1 = c_array, ttumble = ttumble, NBurn = NBurn, dt = dt)

c_data = [res["concentrations"][::tau_sub] for res in results_model_pf]
cmean = np.concatenate(c_data).mean()

c_data = [x/cmax for x in c_data]

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

dir_name = f"../data/model/clevels_cmpf_cmin{cmin}_cmax{cmax}_NLevels{NLevels}_Ntraj{NTraj}_NSteps{NSteps}_ttumble{ttumble}_dt{dt}_tau_sub{tau_sub}"
# check if the directory exists, otherwise create it
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

# save the data
np.savez(dir_name + f"/trajectories_model_clevels_cmin{cmin}_cmax{cmax}_NLevels{NLevels}_Ntraj{NTraj}_NSteps{NSteps}_ttumble{ttumble}_dt{dt}_tau_sub{tau_sub}.npz",
         trajectories_data = trajectories_data, c_array = c_array, ttumble = ttumble, dt = dt,
         NSteps1 = NSteps1, NSteps2 = NSteps2, NBurn = NBurn, tau_sub = tau_sub)

N_FSC = 20
seeds_FSC = np.arange(0, N_FSC)
F = 2
M = 2
A = 2

NEpochs = 20
NBatch = 20
lr = (0.05, 0.05)
gamma = 0.99
train_split = 0.9

for seed in seeds_FSC:
    tic = measure_time.time()
    FSC_tofit = controller.FSC("continuous", M = M, A = A, F = F, seed = seed)

    tloss, vloss = FSC_tofit.fit(trajectories_data, NEpochs = NEpochs,
                                NBatch = NBatch, lr = lr, gamma = gamma, train_split = train_split)

    par_names = dir_name + f"/FSC_M{M}_A{A}_F{F}_model_clevels_cmin{cmin}_cmax{cmax}_NLevels{NLevels}_Ntraj{NTraj}_NSteps{NSteps}_ttumble{ttumble}_dt{dt}_tau_sub{tau_sub}_"
    par_names += f"seed_{seed}_NEpochs{NEpochs}_lr{lr[0]}_{lr[1]}_gamma{gamma}_train_split{train_split}_"

    parameters = FSC_tofit.get_learned_parameters()

    parameters["seed"] = seed
    parameters["tloss"] = tloss
    parameters["vloss"] = vloss
    parameters["NEpochs"] = NEpochs
    parameters["NBatch"] = NBatch
    parameters["lr"] = lr
    parameters["gamma"] = gamma
    parameters["train_split"] = train_split

    np.savez(par_names + "parameters.npz", **parameters)
    toc = measure_time.time()

    print(f"Seed {seed} done in {toc - tic} seconds")
    print()