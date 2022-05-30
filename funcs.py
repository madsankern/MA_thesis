# ----------------------------- #
# Collection of functions I use #
# ----------------------------- #
import numpy as np
from numba import njit

# @njit
def gini(x, w=None):

    '''Compute the gini coefficient for array x'''

    # The rest of the code requires numpy arrays.
    x = np.asarray(x)
    if w is not None:
        w = np.asarray(w)
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = w[sorted_indices]
        # Force float dtype to avoid overflows
        cumw = np.cumsum(sorted_w, dtype=float)
        cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
        return (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) / 
                (cumxw[-1] * cumw[-1]))
    else:
        sorted_x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(sorted_x, dtype=float)
        # The above formula, with all weights equal to 1 simplifies to:
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

def Xi(m,h_underbar):
    '''Function that shows relative gain in toy model'''

    # Set fixed parameters
    alpha = 0.8
    beta = .9
    p = 1.0
    r = 0.03
    dp = 1

    # Numerator
    num = (1 - alpha)*(m/p) - alpha*h_underbar

    # Denominator
    denom = m*(1 - alpha/(alpha+beta) - r*(1-alpha)) - p*h_underbar*(alpha + alpha/(alpha+beta))

    return dp*num / denom
