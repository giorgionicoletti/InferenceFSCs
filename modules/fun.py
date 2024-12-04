import numpy as np
import numba as nb

import torch

def softmax(x, axis = 0):
    """
    Computes the softmax of an array x along a given axis.

    Parameters:
    --- x: np.array
        Array to be softmaxed.
    --- axis: int or tuple of ints
        Axis or axes along which the softmax is computed.

    Returns:
    --- np.array
        Softmaxed array, of the same shape as x.
    """
    max_x = np.max(x, axis = axis, keepdims = True)
    exp_x = np.exp(x - max_x)
    sum_exp_x = np.sum(exp_x, axis = axis, keepdims = True)
    return exp_x / sum_exp_x

def torch_softmax_2dim(x, dims):
    """
    Computes the softmax of a tensor x along a given axis.

    Parameters:
    --- x: torch.Tensor
        Tensor to be softmaxed.
    --- dims: tuple
        Dimensions along which the softmax is computed.

    Returns:
    --- torch.Tensor
        Softmaxed tensor, of the same shape as x.
    """
    max_x = torch.max(x, dim = dims[0], keepdim = True)[0]
    max_x = torch.max(max_x, dim = dims[1], keepdim = True)[0]
    exp_x = torch.exp(x - max_x)
    sum_exp_x = torch.sum(exp_x, dim = dims, keepdim = True)
    return exp_x / sum_exp_x

@nb.njit
def numba_random_choice(vals, probs):
    """
    Chooses a value from vals with probabilities given by probs.

    Parameters:
    --- vals: np.array
        Array of values to choose from.
    --- probs: np.array
        Array of probabilities for each value in vals.

    Returns:
    --- object
        Value chosen from vals.
    """
    r = np.random.rand()
    cum_probs = np.cumsum(probs)
    for idx in range(len(probs)):
        if r < cum_probs[idx]:
            return vals[idx]

def combine_spaces(space1, space2):
    """
    Combines two spaces into a single space. Useful to index the combined
    space with a single index.

    Parameters:
    --- space1: np.array
        First space to be combined.
    --- space2: np.array
        Second space to be combined.

    Returns:
    --- np.array
        Combined space, with shape (space1.size * space2.size, 2).
    """
    return np.array(np.meshgrid(space1, space2)).T.reshape(-1, 2)

import itertools
def get_conditional_permutations_numpy(p_array):
    """
    Get permutations for conditional probability p(a,m'|m,y) from numpy array
    Input array shape: (Y, M, M, A) where
    - First M dimension is m' (output state)
    - Second M dimension is m (input state)
    Returns numpy array of shape (num_perms, Y, M, M, A)
    """
    Y, M, _, A = p_array.shape
    perms = list(itertools.permutations(range(M)))
    
    # Create output array for all permutations
    output = np.zeros((len(perms), A, M, M, Y))
    
    for i, perm in enumerate(perms):
        # Permute both m and m' dimensions according to the same permutation
        permuted = p_array[:, perm, :, :]  # Permute m' (output states)
        permuted = permuted[:, :, perm, :]  # Permute m (input states)
        
        output[i] = permuted
        
    return output



def fun_arg_rho_tilde(rhoVec, wVec):
    return wVec/(wVec*rhoVec).sum(axis = -1)[..., None]

def fun_to_min(psi, wVec, pya):
    rhoVec = np.exp(psi)
    rhoVec /= rhoVec.sum()

    arg_rho_tilde = fun_arg_rho_tilde(rhoVec, wVec)
    
    return np.sum(pya[..., None]*arg_rho_tilde*rhoVec[None, ...], axis = (0,1)) - rhoVec

def fun_MSE(psi, wVec, pya):
    MSE = fun_to_min(psi, wVec, pya)**2
    return np.sum(np.sqrt(MSE))

def fun_jac_rho(rhoVec):
    M = len(rhoVec)
    jac = np.zeros((M, M))
    for mu in range(M):
        for nu in range(M):
            if mu == nu:
                jac[mu, nu] = rhoVec[mu] - rhoVec[mu]*rhoVec[nu]
            else:
                jac[mu, nu] = -rhoVec[mu]*rhoVec[nu]
    
    return jac

def jac_fun(psi, wVec, pya):
    rhoVec = np.exp(psi)
    rhoVec /= rhoVec.sum()

    M = len(rhoVec)

    arg_rho_tilde = fun_arg_rho_tilde(rhoVec, wVec)
    jac_rho = fun_jac_rho(rhoVec)

    jac = np.zeros((M, M))

    for mu in range(M):
        for nu in range(M):
            temp = arg_rho_tilde[:, :, mu]*(jac_rho[mu, nu] + rhoVec[mu]*rhoVec[nu])
            temp -= arg_rho_tilde[:, :, mu]*arg_rho_tilde[:, :, nu]*rhoVec[mu]*rhoVec[nu]
            
            temp = np.sum(pya*temp)

            jac[mu, nu] = temp - jac_rho[mu, nu]
    return jac

def jac_MSE(psi, wVec, pya):
    M = len(psi)
    jac = jac_fun(psi, wVec, pya)
    f = fun_to_min(psi, wVec, pya)
    fsqrt = np.divide(f, np.sqrt(f**2), out=np.zeros_like(f), where=f!=0)
    
    j = np.zeros(M)

    for nu in range(M):
        for mu in range(M):
            j[nu] += fsqrt[mu]*jac[mu, nu]
    return j