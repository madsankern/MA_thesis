import numpy as np
from numba import njit

# Durable
@njit(fastmath=True)
def n_plus_func(d,par):
    n_plus = d # no depreciation
    # n_plus = np.fmin(n_plus,par.n_max) # upper bound, can be removed
    return n_plus

# Cash
@njit(fastmath=True)
def m_plus_func(a,p_plus,par,n):
    y_plus = p_plus
    m_plus = par.R*a + y_plus - par.deltaa*n
    return m_plus

# Cash when adjusting
@njit(fastmath=True)
def x_plus_func(m_plus,n_plus,par):
    return m_plus + par.ph*n_plus

# Add p_buy lom here