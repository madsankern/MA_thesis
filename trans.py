import numpy as np
from numba import njit

# Income process
@njit(fastmath=True)
def p_plus_func(p,psi,par): # LOM for permanent income. Modelled as a Random walk
    p_plus = p*psi
    p_plus = np.fmax(p_plus,par.p_min) # lower bound
    p_plus = np.fmin(p_plus,par.p_max) # upper bound
    return p_plus

# Durable
@njit(fastmath=True)
def n_plus_func(d,par):
    n_plus = d # no depreciation
    n_plus = np.fmin(n_plus,par.n_max) # upper bound
    return n_plus

# Cash - add maintenence cost
@njit(fastmath=True)
def m_plus_func(a,p_plus,par,n):
    y_plus = p_plus # Removed xi_plus
    m_plus = par.R*a + y_plus - par.deltaa*n
    return m_plus

# Cash when adjusting
@njit(fastmath=True)
def x_plus_func(m_plus,n_plus,par):
    return m_plus + (1-par.tau)*par.ph*n_plus # Remove adjustment cost?

# Add p_buy lom here