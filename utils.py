import numpy as np
# from numba import jit

# @jit(nopython=True)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# @jit(nopython=True)
def sigmoid_inv(x):
    return np.log(x / (1-x))


def logsumexp(prob_lst):
    return np.exp(np.sum(np.log(prob_lst)))

# @jit(nopython=True)
def softmax(x):
    x = x - np.max(x)
    x = np.exp(x)
    return x / np.sum(x)

def softmax_entropy(x):
    # x are the logits
    p = softmax(x)
    c = np.max(x)
    logpi = x - np.log(np.sum(np.exp(x - c))) - c
    return -np.sum(p * logpi)


def project_to_probsimplex(x):
    # taken from Chen and Ye (2011)
    # l2 projection to probability simplex
    u = np.array(sorted(x))
    for i in range(len(x)-2, -1, -1):
        t = (np.sum(u[i+1:]) - 1) / (len(x) - 1 - i)
        if t >= u[i]:
            t_hat = t
            break
    else:
        t_hat = (np.sum(u) - 1)/ len(x)
    return np.clip(x - t_hat, 0, 99)  # 99 doesn't matter, it's max(0, x - t_hat)
