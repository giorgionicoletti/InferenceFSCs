import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../new_modules')

import FSC as controller
import torch
import copy

import pandas as pd
import time as measure_time

path = "../data/data_filtered"
df = pd.read_csv(path)
cell_indexes = np.unique(df["particle"])

actions = []
frames = []
observations = []


MinTrLen = 300
MaxTrLen = 800
InitSkip = 10

np.random.seed(0)

for cell_idx in cell_indexes:
    cell = df[df["particle"] == cell_idx]
    cact = cell["tumbling"].values[InitSkip:]
    cfra = cell["frame"].values[InitSkip:]
    cobs = cell["c_norm"].values[InitSkip:]

    # check if a trajectory is shorter than MaxTrLen, otherwise break it into pieces
    if len(cact) < MaxTrLen and len(cact) > MinTrLen:
        actions.append(cact)
        frames.append(cfra)
        observations.append(cobs)
    else:
        NewTrLen = np.random.randint(MinTrLen, MaxTrLen)
        for i in range(0, len(cact), NewTrLen):
            new_traj = cact[i:i+NewTrLen]
            if len(new_traj) < NewTrLen:
                continue
            actions.append(cact[i:i+NewTrLen])
            frames.append(cfra[i:i+NewTrLen])
            observations.append(cobs[i:i+NewTrLen])

NTrajInit = 1000
NTraj = 1000
trajectories = []

for i in range(NTrajInit, NTrajInit + NTraj):
    dict_traj = {}
    dict_traj["actions"] = actions[i].astype(int)
    dict_traj["features"] = np.array([np.ones(observations[i].size).astype(np.float32), observations[i].astype(np.float32)])

    trajectories.append(dict_traj)


first_action = np.array([tr["actions"][0] for tr in trajectories])

print("Number of trajectories: ", len(trajectories))
print("Number of tumbling at the beginning: ", np.sum(first_action))
print("Fraction of tumbling at the beginning: ", np.round(np.sum(first_action) / len(trajectories) * 100, 2), "%")

N_FSC = 20
seeds = np.arange(0, N_FSC)
F = 2
M = 2
A = 2

NEpochs = 10
NBatch = 50
lr = (0.05, 0.05)
gamma = 0.99
train_split = 0.9

for seed in seeds:
    tic = measure_time.time()
    FSC_tofit = controller.FSC("continuous", M = M, A = A, F = F, seed = seed)

    tloss, vloss = FSC_tofit.fit(trajectories, NEpochs = NEpochs, scheduler = "exp",
                                 NBatch = NBatch, lr = lr, gamma = gamma, train_split = train_split)

    par_names = f"../data/parameters/FSC_M{M}_A{A}_F{F}_seed_{seed}_NEpochs{NEpochs}_NTrajs{len(trajectories)}_NTrajInit{NTrajInit}_"
    par_names += f"MinTrLen{MinTrLen}_MaxTrLen{MaxTrLen}_InitSkip{InitSkip}_"

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