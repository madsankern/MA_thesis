import numpy as np
from numba import njit

# from consav import markov

# Durable
@njit(fastmath=True)
def n_plus_func(d,par):
    n_plus = d # no depreciation
    # n_plus = np.fmin(n_plus,par.n_max) # upper bound, can be removed
    return n_plus

# Cash
@njit(fastmath=True)
def m_plus_func(a,y_plus,par,n,R,ph):
    # Added separate interest rate

    m_plus = R*a + y_plus - par.deltaa*n - par.tauc*ph*n
    return m_plus

# Cash when adjusting
@njit(fastmath=True)
def x_plus_func(m_plus,n_plus,pb,par,ph):
    return m_plus + ph*n_plus - par.taug*n_plus*(pb - ph)

