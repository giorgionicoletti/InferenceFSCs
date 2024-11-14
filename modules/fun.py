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
