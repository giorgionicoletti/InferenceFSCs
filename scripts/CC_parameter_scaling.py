import numpy as np

import sys
sys.path.append('../modules')
from FSC import GenerationDiscreteObs as FSC_DiscreteObs

import ObsModels 
import numba as nb
from concurrent.futures import ProcessPoolExecutor

import time as measure_time


@nb.njit
def log_zero(x):
    y = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] != 0:
            y[i] = np.log(x[i])
    return y

@nb.njit
def log_zero_2D(x):
    y = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i, j] != 0:
                y[i, j] = np.log(x[i, j])
    return y

@nb.njit
def optimize_rho(TMat, pya, rhok, maxiter, th = 1e-6):

    Y, M, _, A = TMat.shape

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
        
        rhok_new /= np.sum(rhok_new)

        rhok = rhok_new

    return rhok

def process_trajectory(idx_traj, NTraj, Y_array, A_array, M_array, NStepsObs, MaxIter, seed_traj, seed_np):
    L_local = np.zeros((len(Y_array), len(A_array), len(M_array)))
    DKL_rho_local = np.zeros((len(Y_array), len(A_array), len(M_array)))

    for idx_y, Y in enumerate(Y_array):
        RateMatrixObs = np.ones((Y,Y))
        RateMatrixObs = RateMatrixObs + 2*np.eye(Y)

        observations = ObsModels.DiscreteMarkovChain(NTraj, NStepsObs, RateMatrixObs, initial_seed = seed_traj)

        for idx_a, A in enumerate(A_array):

            for idx_m, M in enumerate(M_array):
                np.random.seed(seed_np)
                
                Theta = np.random.randn(Y, M, M, A)
                Psi = np.random.randn(M)

                FSC = FSC_DiscreteObs(Theta, Psi, verbose = False)
                FSC.load_observations(observations)

                trajectories = FSC.generate_trajectories(NStepsObs)
                ya_array = np.array([[tr['observations'][0], tr['actions'][0]]  for tr in trajectories])
                hist = np.histogram2d(ya_array[:,0], ya_array[:,1], bins = [np.arange(Y+1), np.arange(A+1)])[0]
                pya = hist / np.sum(hist)
                pAgY = pya / np.sum(pya, axis = 1)[:, None]

                rhostart = np.random.rand(M)
                rhostart /= np.sum(rhostart)
                rho_opt = optimize_rho(FSC.TMat, pya, rhostart, MaxIter)

                wVec = FSC.TMat.sum(axis = 2).transpose(0, 2, 1)

                L_local[idx_y, idx_a, idx_m] = np.sum(pya*(log_zero_2D(pAgY) - log_zero_2D(np.sum((wVec*rho_opt), axis = -1))))
                DKL_rho_local[idx_y, idx_a, idx_m] = np.sum(FSC.rho*(log_zero(FSC.rho) - log_zero(rho_opt)))

    return idx_traj, L_local, DKL_rho_local

M_array = np.arange(2, 11, 1)
Y_array = np.arange(2, 6, 1)
A_array = np.arange(2, 6, 1)

NTraj_array = np.array([100, 250, 500, 1000, 2000, 4000, 10000, 20000, 40000, 100000, 200000])

NStepsObs = 1

MaxIter = 20000

NSamples = 250

L = np.zeros((len(Y_array), len(A_array), len(M_array), len(NTraj_array), NSamples))
DKL_rho = np.zeros((len(Y_array), len(A_array), len(M_array), len(NTraj_array), NSamples))

# Generate seeds for reproducibility
np.random.seed(42)
seeds_traj = np.random.randint(0, int(1e7), size=(NSamples, len(NTraj_array)))
seeds_np = np.random.randint(0, int(1e7), size=(NSamples, len(NTraj_array)))

for idx_sample in range(NSamples):
    print(f'Sample {idx_sample+1}/{NSamples}')

    time_start = measure_time.time()

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_trajectory, idx_traj, NTraj, Y_array, A_array, M_array, NStepsObs, MaxIter, seeds_traj[idx_sample, idx_traj], seeds_np[idx_sample, idx_traj])
                   for idx_traj, NTraj in enumerate(NTraj_array)]
        
        for future in futures:
            idx_traj, L_local, DKL_local = future.result()
            L[:, :, :, idx_traj, idx_sample] = L_local
            DKL_rho[:, :, :, idx_traj, idx_sample] = DKL_local

    time_end = measure_time.time()
    print(f'Time elapsed: {time_end - time_start}')
    print()


np.save(f'../data/CC_parameter_scaling_MMax{M_array[-1]}_YMax{Y_array[-1]}_AMax{A_array[-1]}_NSamples{NSamples}_DKL.npy', L)
np.save(f'../data/CC_parameter_scaling_MMax{M_array[-1]}_YMax{Y_array[-1]}_AMax{A_array[-1]}_NSamples{NSamples}_DKL_rho.npy', DKL_rho)
np.save(f'../data/CC_parameter_scaling_MMax{M_array[-1]}_YMax{Y_array[-1]}_AMax{A_array[-1]}_NSamples{NSamples}_NTraj.npy', NTraj_array)
np.save(f'../data/CC_parameter_scaling_MMax{M_array[-1]}_YMax{Y_array[-1]}_AMax{A_array[-1]}_NSamples{NSamples}_M_array.npy', M_array)
np.save(f'../data/CC_parameter_scaling_MMax{M_array[-1]}_YMax{Y_array[-1]}_AMax{A_array[-1]}_NSamples{NSamples}_Y_array.npy', Y_array)
np.save(f'../data/CC_parameter_scaling_MMax{M_array[-1]}_YMax{Y_array[-1]}_AMax{A_array[-1]}_NSamples{NSamples}_A_array.npy', A_array)