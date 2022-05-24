import numpy as np
from numba import njit

# a. Housing state from last period choice
@njit(fastmath=True)
def n_plus_func(d,par):
    n_plus = d
    return n_plus

# b. Cash on hand when keeping
@njit(fastmath=True)
def m_plus_func(a,R,y_plus,n,ph,par):
    m_plus = R*a - par.deltaa*n - par.tauc*ph*n + y_plus
    return m_plus

# c. Cash on hand when adjusting
@njit(fastmath=True)
def x_plus_func(m_plus,n_plus,ph,pb,par):
    return m_plus + (ph*n_plus - (1.0-par.phi)*pb*n_plus) - par.taug*n_plus*(ph - pb) - par.eta