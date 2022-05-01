import numpy as np
from numba import njit

# from consav import markov

# a. Housing state
@njit(fastmath=True)
def n_plus_func(d,par):
    n_plus = d
    return n_plus

# b. Cash on hand when keeping
@njit(fastmath=True)
def m_plus_func(a,y_plus,par,n,R,ph):
    m_plus = R*a + y_plus - par.deltaa*n - par.tauc*ph*n
    return m_plus

# c. Cash on hand when adjusting
@njit(fastmath=True)
def x_plus_func(m_plus,n_plus,pb,par,ph):
    return m_plus + ph*n_plus - par.taug*n_plus*(pb - ph)

