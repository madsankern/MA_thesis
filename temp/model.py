# -*- coding: utf-8 -*-
"""DurableConsumptionModel
Solves a consumption-saving model with a durable consumption good and non-convex adjustment costs with either:
A. vfi: value function iteration (only i C++)
B. nvfi: nested value function iteration (both in Python and C++)
C. negm: nested endogenous grid point method (both in Python and C++)
The do_2d switch turns on the extended version with two durable stocks.
"""

##############
# 1. imports #
##############

import time
import numpy as np

# consav package
from consav import ModelClass, jit # baseline model class and jit
from consav import linear_interp # for linear interpolation
from consav import golden_section_search # for optimization in 1D
from consav.grids import nonlinspace # grids
from consav.quadrature import create_PT_shocks # income shocks

# local modules
import utility
import trans
import last_period
import post_decision
import negm # this is the important one
# import simulate
# import figs

# Define model class
class DurableConsumptionModelClass(ModelClass):
    
    #########
    # setup #
    #########
    
    def settings(self):
        """ choose settings """

        # a. namespaces
        self.namespaces = []
        
        # b. other attributes
        self.other_attrs = []
        
        # c. savefolder - CHECK THIS
        self.savefolder = 'saved'

        # d. list not-floats for safe type inference
        self.not_floats = ['solmethod','T','t','simN','sim_seed','cppthreads',
                           'Npsi','Nxi','Nm','Np','Nn','Nx','Na','Nshocks',
                           'do_2d','do_print','do_print_period','do_marg_u','do_simple_wq']
        
        # e. cpp
        self.cpp_filename = 'cppfuncs/main.cpp'
        self.cpp_options = {'compiler':'vs'}

        # help for linting
        self.cpp.compute_wq_nvfi = None
        self.cpp.compute_wq_nvfi_2d = None
        self.cpp.compute_wq_negm = None
        self.cpp.compute_wq_negm_2d = None

        self.cpp.solve_vfi_keep = None
        self.cpp.solve_vfi_adj = None
        
        self.cpp.solve_vfi_2d_keep = None
        self.cpp.solve_vfi_2d_adj_full = None
        self.cpp.solve_vfi_2d_adj_d1 = None
        self.cpp.solve_vfi_2d_adj_d2 = None

        self.cpp.solve_nvfi_keep = None
        self.cpp.solve_nvfi_adj = None
        
        self.cpp.solve_nvfi_2d_keep = None
        self.cpp.solve_nvfi_2d_adj_full = None
        self.cpp.solve_nvfi_2d_adj_d1 = None
        self.cpp.solve_nvfi_2d_adj_d2 = None

        self.cpp.solve_negm_keep = None
        self.cpp.solve_negm_2d_keep = None

    def setup(self):
        """ set baseline parameters """

        par = self.par

        # a. baseline parameters
        par.do_2d = False

        # horizon
        par.T = 5
        
        # preferences
        par.beta = 0.965
        par.rho = 2.0 # I call this sigma
        par.alpha = 0.9
        par.d_ubar = 1e-2 # Check if this can just be zero

        # returns and income
        par.R = 1.03
        par.tau = 0.10 # Adjustment cost
        par.delta = 0.15 # Depreciation rate
        par.sigma_psi = 0.1 # Shocks to permanent and transitory income below
        par.Npsi = 5 # number of shocks for quadrature (?)
        par.sigma_xi = 0.1
        par.Nxi = 5
        par.pi = 0.0 # What is this?
        par.mu = 0.5
        
        # grids
        par.Np = 50
        par.p_min = 1e-4
        par.p_max = 3.0
        par.Nn = 3 # number of house sizes (change later)
        par.n_min = 1e-6 # smalles house
        par.n_max = 3.0 # largest house
        par.Nm = 100
        par.m_max = 10.0    
        par.Nx = 100
        par.x_max = par.m_max + par.n_max # Adjust this with house price at some point
        par.Na = 100
        par.a_max = par.m_max+1.0

        # simulation
        par.sigma_p0 = 0.2
        par.mu_d0 = 0.8
        par.sigma_d0 = 0.2
        par.mu_a0 = 0.2
        par.sigma_a0 = 0.1
        par.simN = 100000
        par.sim_seed = 1998
        par.euler_cutoff = 0.02

        # misc - see if this can be deleted
        par.solmethod = 'nvfi'
        par.t = 0
        par.tol = 1e-8
        par.do_print = False
        par.do_print_period = False
        par.cppthreads = 8
        par.do_simple_wq = False # not using optimized interpolation in C++
        par.do_marg_u = False # calculate marginal utility for use in egm
        
    def allocate(self):
        """ allocate model, i.e. create grids and allocate solution and simluation arrays """

        self.create_grids()
        self.solve_prep()
        self.simulate_prep()
            
    def create_grids(self):
        """ construct grids for states and shocks """
        
        par = self.par

        # This will always hold for me
        if par.solmethod in ['negm','negm_cpp','negm_2d_cpp']:
            par.do_marg_u = True

        # a. states        
        par.grid_p = nonlinspace(par.p_min,par.p_max,par.Np,1.1) # Permanent income state
        par.grid_n = np.linspace(par.n_min,par.n_max,par.Nn) # This must be a linear grid
        par.grid_m = nonlinspace(0,par.m_max,par.Nm,1.1)
        par.grid_x = nonlinspace(0,par.x_max,par.Nx,1.1)
        
        # b. post-decision states
        par.grid_a = nonlinspace(0,par.a_max,par.Na,1.1)
        
        # c. shocks for income process
        shocks = create_PT_shocks(
            par.sigma_psi,par.Npsi,par.sigma_xi,par.Nxi,
            par.pi,par.mu)
        par.psi,par.psi_w,par.xi,par.xi_w,par.Nshocks = shocks

        # d. set seed
        np.random.seed(par.sim_seed)

        # e. timing - not needed for me
        par.time_w = np.zeros(par.T)
        par.time_keep = np.zeros(par.T)
        par.time_adj = np.zeros(par.T)
        par.time_adj_full = np.zeros(par.T)
        par.time_adj_d1 = np.zeros(par.T)
        par.time_adj_d2 = np.zeros(par.T)

    def checksum(self,simple=False,T=1):
        """ calculate and print checksum """

        par = self.par
        sol = self.sol

        if simple:
            if par.do_2d:
                print(f'checksum, inv_v_keep: {np.mean(sol.inv_v_keep_2d[0]):.8f}')
                print(f'checksum, inv_v_adj_full: {np.mean(sol.inv_v_adj_full_2d[0]):.8f}')
                print(f'checksum, inv_v_adj_d1_2d: {np.mean(sol.inv_v_adj_d1_2d[0]):.8f}')
                print(f'checksum, inv_v_adj_d2_2d: {np.mean(sol.inv_v_adj_d2_2d[0]):.8f}')
            else:
                print(f'checksum, inv_v_keep: {np.mean(sol.inv_v_keep[0]):.8f}')
                print(f'checksum, inv_v_adj: {np.mean(sol.inv_v_adj[0]):.8f}')
            return

        print('')
        for t in range(T):
            if par.do_2d:
                
                if t < par.T-1:
                    print(f'checksum, inv_w: {np.mean(sol.inv_w_2d[t]):.8f}')
                    print(f'checksum, q: {np.mean(sol.q_2d[t]):.8f}')

                print(f'checksum, inv_v_keep: {np.mean(sol.inv_v_keep_2d[t]):.8f}')
                print(f'checksum, inv_v_adj_full: {np.mean(sol.inv_v_adj_full_2d[t]):.8f}')
                print(f'checksum, inv_v_adj_d1_2d: {np.mean(sol.inv_v_adj_d1_2d[t]):.8f}')
                print(f'checksum, inv_v_adj_d2_2d: {np.mean(sol.inv_v_adj_d2_2d[t]):.8f}')
                print(f'checksum, inv_marg_u_keep: {np.mean(sol.inv_marg_u_keep_2d[t]):.8f}')
                print(f'checksum, inv_marg_u_adj_full: {np.mean(sol.inv_marg_u_adj_full_2d[t]):.8f}')
                print(f'checksum, inv_marg_u_adj_d1_2d: {np.mean(sol.inv_marg_u_adj_d1_2d[t]):.8f}')
                print(f'checksum, inv_marg_u_adj_d2_2d: {np.mean(sol.inv_marg_u_adj_d2_2d[t]):.8f}')                
                print('')

            else:

                if t < par.T-1:
                    print(f'checksum, inv_w: {np.mean(sol.inv_w[t]):.8f}')
                    print(f'checksum, q: {np.mean(sol.q[t]):.8f}')

                print(f'checksum, c_keep: {np.mean(sol.c_keep[t]):.8f}')
                print(f'checksum, d_adj: {np.mean(sol.d_adj[t]):.8f}')
                print(f'checksum, c_adj: {np.mean(sol.c_adj[t]):.8f}')
                print(f'checksum, inv_v_keep: {np.mean(sol.inv_v_keep[t]):.8f}')
                print(f'checksum, inv_marg_u_keep: {np.mean(sol.inv_marg_u_keep[t]):.8f}')
                print(f'checksum, inv_v_adj: {np.mean(sol.inv_v_adj[t]):.8f}')
                print(f'checksum, inv_marg_u_adj: {np.mean(sol.inv_marg_u_adj[t]):.8f}')
                print('')
                # not sure about this...
    #########
    # solve #
    #########

    def precompile_numba(self):
        """ solve the model with very coarse grids and simulate with very few persons"""

        par = self.par

        tic = time.time()

        # a. define
        fastpar = dict()
        fastpar['do_print'] = False
        fastpar['do_print_period'] = False
        fastpar['T'] = 2
        fastpar['Np'] = 3
        fastpar['Nn'] = par.Nn # Just set this as the same for simplicity
        fastpar['Nm'] = 3
        fastpar['Nx'] = 3
        fastpar['Na'] = 3
        fastpar['simN'] = 2

        # b. apply
        for key,val in fastpar.items():
            prev = getattr(par,key)
            setattr(par,key,val)
            fastpar[key] = prev

        self.allocate()

        # c. solve
        self.solve(do_assert=False)

        # d. simulate
        self.simulate()

        # e. reiterate
        for key,val in fastpar.items():
            setattr(par,key,val)
        
        self.allocate()

        toc = time.time()
        if par.do_print:
            print(f'numba precompiled in {toc-tic:.1f} secs')

    
    # Prepare model to be solved - allocates memory
    def solve_prep(self):
        """ allocate memory for solution """

        # Define parameter and solution methods
        par = self.par
        sol = self.sol

        # a. standard - not 2d model
        if not par.do_2d: keep_shape = (par.T,par.Np,par.Nn,par.Nm)
        else: keep_shape = (0,0,0,0) # Check what this is
        
        sol.c_keep = np.zeros(keep_shape)
        sol.inv_v_keep = np.zeros(keep_shape)
        sol.inv_marg_u_keep = np.zeros(keep_shape)

        if not par.do_2d: adj_shape = (par.T,par.Np,par.Nx)
        else: adj_shape = (0,0,0)
        sol.d_adj = np.zeros(adj_shape)
        sol.c_adj = np.zeros(adj_shape)
        sol.inv_v_adj = np.zeros(adj_shape)
        sol.inv_marg_u_adj = np.zeros(adj_shape)
            
        if not par.do_2d: post_shape = (par.T-1,par.Np,par.Nn,par.Na)
        else: post_shape = (0,0,0,0)
        sol.inv_w = np.nan*np.zeros(post_shape)
        sol.q = np.nan*np.zeros(post_shape)
        sol.q_c = np.nan*np.zeros(post_shape) # check what these two are
        sol.q_m = np.nan*np.zeros(post_shape)

    # Function to solve the model
    def solve(self,do_assert=True):
        """ solve the model
        
        Args:
            do_assert (bool,optional): make assertions on the solution
        
        """
        # if set to 2d - I do not do this!
        if self.par.do_2d: return self.solve_2d()
        cpp = self.cpp

        tic = time.time()
            
        # backwards induction - Loop backwards from T
        for t in reversed(range(self.par.T)):
            
            self.par.t = t

            # Check what this is
            with jit(self) as model:

                par = model.par
                sol = model.sol
                
                # i. last period
                if t == par.T-1:

                    # Use last period solver
                    last_period.solve(t,sol,par)

                    # Check that everything makes sense
                    if do_assert:
                        assert np.all((sol.c_keep[t] >= 0) & (np.isnan(sol.c_keep[t]) == False))
                        assert np.all((sol.inv_v_keep[t] >= 0) & (np.isnan(sol.inv_v_keep[t]) == False))
                        assert np.all((sol.d_adj[t] >= 0) & (np.isnan(sol.d_adj[t]) == False))
                        assert np.all((sol.c_adj[t] >= 0) & (np.isnan(sol.c_adj[t]) == False))
                        assert np.all((sol.inv_v_adj[t] >= 0) & (np.isnan(sol.inv_v_adj[t]) == False))

                # ii. all other periods
                else:
                    
                    # o. compute post-decision functions
                    tic_w = time.time()

                    if par.solmethod in ['negm']:
                        post_decision.compute_wq(t,sol,par,compute_q=True)

                    toc_w = time.time()
                    par.time_w[t] = toc_w-tic_w
                    if par.do_print:
                        print(f'  w computed in {toc_w-tic_w:.1f} secs')

                    if do_assert and par.solmethod in ['nvfi','negm']:
                        assert np.all((sol.inv_w[t] > 0) & (np.isnan(sol.inv_w[t]) == False)), t 
                        if par.solmethod in ['negm']:                                                       
                            assert np.all((sol.q[t] > 0) & (np.isnan(sol.q[t]) == False)), t

                    # oo. solve keeper problem
                    tic_keep = time.time()
                    negm.solve_keep(t,sol,par)

                    toc_keep = time.time()
                    par.time_keep[t] = toc_keep-tic_keep
                    if par.do_print:
                        print(f'  solved keeper problem in {toc_keep-tic_keep:.1f} secs')

                    if do_assert:
                        assert np.all((sol.c_keep[t] >= 0) & (np.isnan(sol.c_keep[t]) == False)), t
                        assert np.all((sol.inv_v_keep[t] >= 0) & (np.isnan(sol.inv_v_keep[t]) == False)), t

                    # ooo. solve adjuster problem
                    tic_adj = time.time()
                    
                    elif par.solmethod in ['nvfi','negm']:
                        nvfi.solve_adj(t,sol,par) # This is using the nvfi code as the step is the same for negm                   
                    elif par.solmethod == 'vfi_cpp':
                        cpp.solve_vfi_adj(par,sol)  
                    elif par.solmethod in ['nvfi_cpp','negm_cpp']:
                        cpp.solve_nvfi_adj(par,sol)  

                    toc_adj = time.time()
                    par.time_adj[t] = toc_adj-tic_adj
                    if par.do_print:
                        print(f'  solved adjuster problem in {toc_adj-tic_adj:.1f} secs')

                    if do_assert:
                        assert np.all((sol.d_adj[t] >= 0) & (np.isnan(sol.d_adj[t]) == False)), t
                        assert np.all((sol.c_adj[t] >= 0) & (np.isnan(sol.c_adj[t]) == False)), t
                        assert np.all((sol.inv_v_adj[t] >= 0) & (np.isnan(sol.inv_v_adj[t]) == False)), t

                # iii. print
                toc = time.time()
                if par.do_print or par.do_print_period:
                    print(f' t = {t} solved in {toc-tic:.1f} secs')