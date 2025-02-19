import numpy as np
import numba as nb
import fun

@nb.njit
def ContinuousDichotomousObservations(NSteps, w0, w1, dt):
    wsum = w0 + w1
    p_stay_0 = w0/wsum + w1/wsum*np.exp(-dt*wsum)
    p_to_0 = w1/wsum - w0/wsum*np.exp(-dt*wsum)
    
    dich = np.zeros(NSteps, dtype = np.int8)
    dich[0] = np.random.choice([0, 1])
    
    for idx in range(NSteps-1):
        R = np.random.rand()
        if dich[idx] == 0:
            if R < p_stay_0:
                dich[idx+1] = 0
            else:
                dich[idx+1] = 1
        if dich[idx] == 1:
            if R < p_to_0:
                dich[idx+1] = 0
            else:
                dich[idx+1] = 1
    
    return dich

@nb.njit
def SingleDiscreteMarkovChain(NSteps, RateMatrix, initial_state = None, seed = None):
    if seed is not None:
        np.random.seed(seed)

    NStates = RateMatrix.shape[0]
    StateSpace = np.arange(NStates)

    states = np.zeros(NSteps, dtype = np.int32)
    if initial_state is None:
        states[0] = np.random.choice(NStates)
    else:
        states[0] = initial_state

    TransitionProbabilities = np.zeros((NStates, NStates))

    for idx in range(NStates):
        TransitionProbabilities[idx] = RateMatrix[idx] / RateMatrix[idx].sum()
    
    for idx in range(1, NSteps):
        states[idx] = fun.numba_random_choice(StateSpace, TransitionProbabilities[states[idx-1]])

    return states

@nb.njit
def DiscreteMarkovChain(NTraj, NSteps, RateMatrix, initial_state = None, initial_seed = None):    
    states = np.zeros((NTraj, NSteps), dtype = np.int32)
    
    for idx in nb.prange(NTraj):
        if initial_seed is not None:
            seed = initial_seed + idx
        else:
            seed = None
        states[idx] = SingleDiscreteMarkovChain(NSteps, RateMatrix, initial_state, seed = seed)
    
    return states


@nb.njit
def get_waiting_times(N, rate):
    x = np.random.rand(N)

    return -np.log(1-x)/rate

@nb.njit
def SingleLinearRamps(NSteps, rate, mplus, mminus, dt, y0 = None, seed = None):
    obs = np.zeros(NSteps, dtype = np.float32)

    if seed is not None:
        np.random.seed(seed)

    if y0 is None:
        y0 = np.random.rand()
    
    obs[0] = y0
    current_m_idx = 0

    mlist = [mplus, mminus]

    for idx in range(1, NSteps):
        if np.random.rand() < rate*dt:
            current_m_idx = 1 - current_m_idx

        obs[idx] = obs[idx-1] + mlist[current_m_idx]*dt

        if obs[idx] < 0:
            current_m_idx = 0
            obs[idx] = obs[idx-1] + mlist[current_m_idx]*dt

    return obs

@nb.njit(parallel = True)
def LinearRamps(NTraj, NSteps, rate, mplus, mminus, dt, initial_obs = None, initial_seed = None):
    obs = np.zeros((NTraj, NSteps), dtype = np.float32)

    for idx in nb.prange(NTraj):
        if initial_seed is not None:
            seed = initial_seed + idx
        obs[idx] = SingleLinearRamps(NSteps, rate, mplus, mminus, dt, initial_obs, seed = seed)

    return obs


