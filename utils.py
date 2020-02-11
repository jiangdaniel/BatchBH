import pickle
from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy.special import factorial


def bh(pvals, alpha):
    sorted_indices = np.argsort(pvals)
    c = alpha / pvals.size * np.arange(1, pvals.size+1)  # critical values
    reject = np.where(pvals[sorted_indices] <= c)[0]

    if reject.size > 0:
        num_rejects = reject.max()+1
        p_threshold = c[num_rejects-1]
    else:
        num_rejects = 0
        p_threshold = 0
    return num_rejects, p_threshold


def storey_bh(pvals, alpha, lmbda):
    '''LMBDA is candidacy threshold'''
    pi0 = (1 + (pvals > lmbda).sum()) / (pvals.size * (1-lmbda))
    pvals_sorted = pvals.copy()
    pvals_sorted.sort()
    num_rejects = 0
    p_threshold = 0
    for i, p in enumerate(pvals_sorted, start=1):
        fdp_hat = pvals.size * pi0 * p / i
        if fdp_hat > alpha:
            break
        else:
            p_threshold = p
            num_rejects = i
    return num_rejects, p_threshold


GAMMAS_UNNORMALIZED = {
    'log': lambda x: np.log(np.clip(x, 2, None)) / (x * np.exp(np.sqrt(np.log(x)))),
    'poly': lambda x: x ** (-2),
    'exp': lambda x: (1/2) ** x,
    'factorial': lambda x: 1 / factorial(x),
    'super': lambda x: (1 / x) ** x,
    'half': lambda x: (x < 3) / 2
}

GAMMA_COEFFICIENTS = {}
for k, func in GAMMAS_UNNORMALIZED.items():
    GAMMA_COEFFICIENTS[k] = 1 / func(np.arange(1, 1e7)).sum()


def gamma(name, x):
    return GAMMAS_UNNORMALIZED[name](x) * GAMMA_COEFFICIENTS[name]


def timestamp():
    return datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M-%S')


def transpose(lst):
    '''Transpose a 2D list.'''
    return list(map(list, zip(*lst)))


def zero_lvl_defaultdict():
    return dict()

def one_lvl_defaultdict():
    return defaultdict(zero_lvl_defaultdict)

def two_lvl_defaultdict():
    return defaultdict(one_lvl_defaultdict)


def save(fig, fpath):
    fig.savefig(fpath.with_suffix('.png'))
    with fpath.with_suffix('.pkl').open('wb') as f:
        pickle.dump(fig, f)
