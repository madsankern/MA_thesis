# -*- coding: utf-8 -*-
"""DurableConsumptionModel

Solve a consumption-saving model with a non-durable numeraire and durable housing

"""

##############
# 1. imports #
##############

import time
import numpy as np

# consav package
from consav import ModelClass, jit # baseline model class and jit
from consav import linear_interp # for linear interpolation
from consav.grids import nonlinspace # grids

# local modules
import utility
import trans
import last_period
import post_decision
import nvfi
import negm
import simulate

import path
import simulate_path


class DurableConsumptionModelClass(ModelClass): # Rename
    
    #########
    # Setup #
    #########
    
    def settings(self):
        """ choose settings """

        # a. namespaces
        self.namespaces = ['sol_path', 'sim_path']
        
        # b. other attributes
        self.other_attrs = ['']
        
        # c. savefolder
        self.savefolder = 'saved'

        # d. list not-floats for safe type inference - UPDATE
        self.not_floats = ['solmethod','T','t','simN','sim_seed','cppthreads',
                           'Npsi','Nxi','Nm','Np','Nn','Nx','Na','Nshocks',
                           'do_print','do_print_period','do_marg_u','do_simple_wq']

    def setup(self):
        """ set baseline parameters """

        par = self.par

        # a. Horizon
        par.T = 50 # Number of iterations to find stationary solution
        par.path_T = 300 # 300 periods AFTER the shock occured
        par.sim_T = 200 # Length of stationary simulatin to ensure convergence
        
        # b. Preferences
        par.beta = 0.965
        par.rho = 2.0
        par.alpha = 0.9
        par.d_ubar = 0.2

        # c. Prices and costs
        par.R = 1.03
        par.ph = 6.247127525452337 # House price - rename to p, set to equilibrium
        par.tau = 0.10 # can be removed
        par.deltaa = 0.15 # maintenence cost

        # d. Path for aggregate states
        par.path_R = np.full(par.sim_T + par.path_T, par.R) # for impulse response
        par.path_ph = np.full(par.sim_T + par.path_T, par.ph) # House price sequence

        par.pi = 0.0 # what is this
        par.mu = 0.5 # what is this

        # e. Markov process income
        par.p_12 = 0.33
        par.p_21 = 0.33

        par.p_mat = np.array([ 
            [1-par.p_12, par.p_12], 
            [par.p_21, 1-par.p_21]])
        par.p_mat_cum = np.array([np.cumsum(par.p_mat[i,:]) for i in range(2)])

        par.pi = np.array([1/2,1/2]) # stationary distribution
        par.pi_cum = np.array(np.cumsum(par.pi))

        # f. Purchase price
        par.Npb = 2 # points in the grid
        par.pb_max = 5.0 # max value
        par.pb_min = 0.1 # min value
        
        # g. Taxes
        par.tauc = 0.0 # Wealth tax
        par.taug = 0.0 # Gains tax

        # h. Grids
        par.Ny = 2 #6 # update this
        par.y_min = 0.3
        par.y_max = 1.0
        par.Nn = 10
        par.n_max = 0.15
        par.Nm = 100
        par.m_max = 10.0    
        par.Nx = 100
        par.x_max = par.m_max + par.n_max # Add house price
        par.Na = 100
        par.a_max = par.m_max+1.0

        # i. Simulation - these must be based on steady state
        par.sigma_p0 = 0.2
        par.mu_d0 = 0.8
        par.sigma_d0 = 0.2
        par.mu_a0 = 0.6
        par.sigma_a0 = 0.1 # variance of initial assets
        par.simN = 100_000
        par.sim_seed = 1998
        par.euler_cutoff = 0.02

        # j. Misc
        par.solmethod = 'negm' # this might also be redundant
        par.t = 0
        par.tol = 1e-6
        par.do_print = False
        par.do_print_period = False
        par.cppthreads = 8
        par.do_simple_wq = False # not using optimized interpolation in C++
        par.do_marg_u = False # calculate marginal utility for use in egm
        
    def allocate(self):
        """ allocate model, i.e. create grids and allocate solution and simluation arrays """

        self.create_grids()

        # a. Steady state
        self.solve_prep()
        self.simulate_prep()
        
        # b. Transition path
        self.solve_path_prep()
        self.simulate_path_prep()
            
    def create_grids(self):
        """ construct grids for states and shocks """
        
        par = self.par
        sim = self.sim
        sim_path = self.sim_path

        par.do_marg_u = True

        # a. States
        par.grid_y = nonlinspace(par.y_min,par.y_max,par.Ny,1.1)
        par.grid_n = nonlinspace(0,par.n_max,par.Nn,1.1)
        par.grid_m = nonlinspace(0,par.m_max,par.Nm,1.1)
        par.grid_x = nonlinspace(0,par.x_max,par.Nx,1.1)
        par.grid_pb = nonlinspace(par.pb_min,par.pb_max,par.Npb,1.1)
        
        # b. Post-decision states
        par.grid_a = nonlinspace(0,par.a_max,par.Na,1.1)

        # d. Set seed
        np.random.seed(par.sim_seed)

        # e. Timing
        par.time_w = np.zeros(par.T)
        par.time_keep = np.zeros(par.T)
        par.time_adj = np.zeros(par.T)
        par.time_adj_full = np.zeros(par.T)

        # f. Matrix of income shocks
        sim.rand = np.zeros(shape=(par.sim_T,par.simN))
        sim_path.rand = np.zeros(shape=(par.path_T+par.sim_T,par.simN))
        sim.rand[:,:] = np.random.uniform(size=(par.sim_T,par.simN))
        sim_path.rand[:,:] = np.random.uniform(size=(par.path_T+par.sim_T,par.simN))

    def checksum(self,simple=False,T=1): # update
        """ calculate and print checksum """

        par = self.par
        sol = self.sol

        if simple:
            print(f'checksum, inv_v_keep: {np.mean(sol.inv_v_keep[0]):.8f}')
            print(f'checksum, inv_v_adj: {np.mean(sol.inv_v_adj[0]):.8f}')
            return

        print('')
        for t in range(T):
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

    #########
    # solve #
    #########

    def precompile_numba(self):
        """ solve the model with very coarse grids and simulate with very few persons"""

        par = self.par

        tic = time.time()

        # a. Define
        fastpar = dict()
        fastpar['do_print'] = False
        fastpar['do_print_period'] = False
        fastpar['T'] = 2
        fastpar['sim_T'] = 2
        fastpar['path_T'] = 2
        fastpar['Ny'] = 2
        fastpar['Nn'] = 3
        fastpar['Nm'] = 3
        fastpar['Nx'] = 3
        fastpar['Na'] = 3
        fastpar['simN'] = 2

        # b. Apply
        for key,val in fastpar.items():
            prev = getattr(par,key)
            setattr(par,key,val)
            fastpar[key] = prev

        self.allocate()

        # c. Solve
        self.solve(do_assert=False)

        # d. Simulate
        self.simulate()

        # Add solve_path and simulate_path - check why this seems to be wrong
        # self.solve_path()
        # self.simulate_path()

        # e. Reiterate
        for key,val in fastpar.items():
            setattr(par,key,val)

        self.allocate()

        toc = time.time()
        if par.do_print:
            print(f'numba precompiled in {toc-tic:.1f} secs')

    def solve_prep(self):
        """ allocate memory for solution """

        par = self.par
        sol = self.sol

        # a. Keeper
        keep_shape = (par.T,par.Npb,par.Ny,par.Nn,par.Nm)
        sol.c_keep = np.zeros(keep_shape)
        sol.inv_v_keep = np.zeros(keep_shape)
        sol.inv_marg_u_keep = np.zeros(keep_shape)

        # b. Adjuster
        adj_shape = (par.T,par.Npb,par.Ny,par.Nx)
        sol.d_adj = np.zeros(adj_shape)
        sol.c_adj = np.zeros(adj_shape)
        sol.inv_v_adj = np.zeros(adj_shape)
        sol.inv_marg_u_adj = np.zeros(adj_shape)
            
        # c. Post decision
        post_shape = (par.T-1,par.Npb,par.Ny,par.Nn,par.Na)
        sol.inv_w = np.nan*np.zeros(post_shape)
        sol.q = np.nan*np.zeros(post_shape)
        sol.q_c = np.nan*np.zeros(post_shape)
        sol.q_m = np.nan*np.zeros(post_shape)

        # d. Distance between iterations
        sol.dist = np.nan*np.zeros(par.T)

    def solve(self,do_assert=True):
        """ solve the model
        
        Args:
            do_assert (bool,optional): make assertions on the solution
        
        """

        tic = time.time()

        for t in reversed(range(self.par.T)):
            
            self.par.t = t

            with jit(self) as model:

                par = model.par
                sol = model.sol
                
                # i. last period
                if t == par.T-1:

                    last_period.solve(t,sol,par,par.ph)

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
                    
                    post_decision.compute_wq(t,par.R,sol,par,par.ph,compute_q=True)

                    toc_w = time.time()
                    par.time_w[t] = toc_w-tic_w
                    if par.do_print:
                        print(f'  w computed in {toc_w-tic_w:.1f} secs')

                    assert np.all((sol.inv_w[t] > 0) & (np.isnan(sol.inv_w[t]) == False)), t 
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
                    
                    nvfi.solve_adj(t,sol,par,par.ph)                  

                    toc_adj = time.time()
                    par.time_adj[t] = toc_adj-tic_adj
                    if par.do_print:
                        print(f'  solved adjuster problem in {toc_adj-tic_adj:.1f} secs')

                    if do_assert:
                        assert np.all((sol.d_adj[t] >= 0) & (np.isnan(sol.d_adj[t]) == False)), t
                        assert np.all((sol.c_adj[t] >= 0) & (np.isnan(sol.c_adj[t]) == False)), t
                        assert np.all((sol.inv_v_adj[t] >= 0) & (np.isnan(sol.inv_v_adj[t]) == False)), t

                # Compute distance to previous iteration
                if t < par.T-1:
                    # sol.dist[t] = np.abs(np.max(sol.c_keep[t+1,:,:,:,:] - sol.c_keep[t,:,:,:,:]))
                    sol.dist[t] = np.abs(np.max(sol.c_adj[t+1,:,:,:] - sol.c_adj[t,:,:,:]))

                # iii. print
                toc = time.time()
                if par.do_print or par.do_print_period:
                    print(f' t = {t} solved in {toc-tic:.1f} secs')

    ############
    # simulate #
    ############

    def simulate_prep(self):
        """ allocate memory for simulation """

        par = self.par
        sim = self.sim

        # a. Initial and final
        sim.p0 = np.zeros(par.simN)
        sim.d0 = np.zeros(par.simN)
        sim.a0 = np.zeros(par.simN)

        # b. Household utility
        sim.utility = np.zeros(par.simN)

        # c. States and choices
        sim_shape = (par.sim_T,par.simN)
        sim.y = np.zeros(sim_shape)
        sim.m = np.zeros(sim_shape)
        sim.n = np.zeros(sim_shape)
        sim.discrete = np.zeros(sim_shape,dtype=np.int)
        sim.d = np.zeros(sim_shape)
        sim.c = np.zeros(sim_shape)
        sim.a = np.zeros(sim_shape)
        
        # d. Euler errors
        euler_shape = (par.sim_T-1,par.simN)
        sim.euler_error = np.zeros(euler_shape)
        sim.euler_error_c = np.zeros(euler_shape)
        sim.euler_error_rel = np.zeros(euler_shape)

        # e. Income states
        sim.state = np.zeros((par.sim_T,par.simN),dtype=np.int_)

    def simulate(self,do_utility=False,do_euler_error=False):
        """ simulate the model """

        par = self.par
        sol = self.sol
        sim = self.sim

        tic = time.time()

        # a. Random initial allocation of housing and cash on hand
        sim.d0[:] = np.random.choice(par.grid_n,size=par.simN) # Initial housing (discrete values)
        sim.a0[:] = par.mu_a0*np.random.lognormal(mean=1.3,sigma=par.sigma_a0,size=par.simN) # initial cash on hand

        # b. call
        with jit(self) as model:

            par = model.par
            sol = model.sol
            sim = model.sim

            simulate.lifecycle(sim,sol,par,path=False)

        toc = time.time()
        
        if par.do_print:
            print(f'model simulated in {toc-tic:.1f} secs')


    #########################
    # Solve Transition Path #
    #########################

    def solve_path_prep(self):
        """ allocate memory for solution along a transition path"""

        par = self.par
        sol_path = self.sol_path

        # a. Keeper
        keep_shape_path = (par.sim_T + par.path_T,par.Npb,par.Ny,par.Nn,par.Nm)
        sol_path.c_keep = np.zeros(keep_shape_path)
        sol_path.inv_v_keep = np.zeros(keep_shape_path)
        sol_path.inv_marg_u_keep = np.zeros(keep_shape_path)

        # b. Adjuster
        adj_shape_path = (par.sim_T + par.path_T,par.Npb,par.Ny,par.Nx)
        sol_path.d_adj = np.zeros(adj_shape_path)
        sol_path.c_adj = np.zeros(adj_shape_path)
        sol_path.inv_v_adj = np.zeros(adj_shape_path)
        sol_path.inv_marg_u_adj = np.zeros(adj_shape_path)

        # c. Post decision
        post_shape_path = (par.sim_T + par.path_T-1,par.Npb,par.Ny,par.Nn,par.Na)
        sol_path.inv_w = np.nan*np.zeros(post_shape_path)
        sol_path.q = np.nan*np.zeros(post_shape_path)
        sol_path.q_c = np.nan*np.zeros(post_shape_path)
        sol_path.q_m = np.nan*np.zeros(post_shape_path)


    def solve_path(self): # Add options for type of shock
        '''Solve the household problem along a transition path'''

        # a. Generate exogenous path of interest rates
        path.gen_path_R(self.par)

        for t in reversed(range(self.par.sim_T, self.par.sim_T + self.par.path_T)):

            with jit(self) as model:

                par = self.par
                sol_path = self.sol_path

                # i. Update interest rates and house prices from path
                R = par.path_R[t]
                ph = par.path_ph[t]
                
                # ii. Last period
                if t == (par.path_T+par.sim_T)-1:

                    last_period.solve(t,sol_path,par,ph) # modify this with the stationary solution

                else:
                    
                    # o. Compute post decision value
                    post_decision.compute_wq(t,R,sol_path,par,ph,compute_q=True) # Only need to update r here, then adj and keep should be okay

                    # oo. Solve keeper
                    negm.solve_keep(t,sol_path,par)

                    # ooo. Solve adjuster
                    nvfi.solve_adj(t,sol_path,par,ph)

        # b. Append the solution to the initial steady state
        with jit(self) as model:

            par = self.par
            sol_path = self.sol_path
            sol = self.sol

            # i. Keeper
            sol_path.c_keep[0:par.sim_T] = sol.c_keep[0]
            sol_path.inv_v_keep[0:par.sim_T] = sol.inv_v_keep[0]
            sol_path.inv_marg_u_keep[0:par.sim_T] = sol.inv_marg_u_keep[0]

            # ii. Adjuster
            sol_path.d_adj[0:par.sim_T] = sol.d_adj[0]
            sol_path.c_adj[0:par.sim_T] = sol.c_adj[0]
            sol_path.inv_v_adj[0:par.sim_T] = sol.inv_v_adj[0]
            sol_path.inv_marg_u_adj[0:par.sim_T] = sol.inv_marg_u_adj[0]
                
            # iii. Post decision
            sol_path.inv_w[0:par.sim_T] = sol.inv_w[0]
            sol_path.q[0:par.sim_T] = sol.q[0]
            sol_path.q_c[0:par.sim_T] = sol.q_c[0]
            sol_path.q_m[0:par.sim_T] = sol.q[0]

        # c. Replace end points with the 50'th iteration
        with jit(self) as model:

            par = self.par
            sol_path = self.sol_path
            sol = self.sol

            path_length = 50

            # i. Keeper
            sol_path.c_keep[par.sim_T+path_length:] = sol_path.c_keep[par.sim_T+path_length]
            sol_path.inv_v_keep[par.sim_T+path_length:] = sol_path.inv_v_keep[par.sim_T+path_length]
            sol_path.inv_marg_u_keep[par.sim_T+path_length:] = sol_path.inv_marg_u_keep[par.sim_T+path_length]

            # ii. Adjuster
            sol_path.d_adj[par.sim_T+path_length:] = sol_path.d_adj[par.sim_T+path_length]
            sol_path.c_adj[par.sim_T+path_length:] = sol_path.c_adj[par.sim_T+path_length]
            sol_path.inv_v_adj[par.sim_T+path_length:] = sol_path.inv_v_adj[par.sim_T+path_length]
            sol_path.inv_marg_u_adj[par.sim_T+path_length:] = sol_path.inv_marg_u_adj[par.sim_T+path_length]
                
            # iii. Post decision
            sol_path.inv_w[par.sim_T+path_length:] = sol_path.inv_w[par.sim_T+path_length]
            sol_path.q[par.sim_T+path_length:] = sol_path.q[par.sim_T+path_length]
            sol_path.q_c[par.sim_T+path_length:] = sol_path.q_c[par.sim_T+path_length]
            sol_path.q_m[par.sim_T+path_length:] = sol_path.q[par.sim_T+path_length]       

    ############################
    # Simulate Transition Path #
    ############################
    
    def simulate_path_prep(self):
        """ allocate memory for simulation """

        par = self.par
        sim_path = self.sim_path

        # a. initial and final
        sim_path.p0 = np.zeros(par.simN)
        sim_path.d0 = np.zeros(par.simN)
        sim_path.a0 = np.zeros(par.simN)

        # b. Household utility
        sim_path.utility = np.zeros(par.simN)

        # c. States and choices
        sim_shape_path = (par.sim_T + 50,par.simN)
        sim_path.y = np.zeros(sim_shape_path)
        sim_path.m = np.zeros(sim_shape_path)
        sim_path.n = np.zeros(sim_shape_path)
        sim_path.discrete = np.zeros(sim_shape_path,dtype=np.int)
        sim_path.d = np.zeros(sim_shape_path)
        sim_path.c = np.zeros(sim_shape_path)
        sim_path.a = np.zeros(sim_shape_path)

        # d. Income states
        sim_path.state = np.zeros((par.sim_T + 50,par.simN),dtype=np.int_)  # sim_shape_path?

    def simulate_path(self):
        '''Simulate a panel of households along a transition path'''
    
        par = self.par
        sol_path = self.sol_path
        sim_path = self.sim_path

        # a. Random initial allocation
        sim_path.d0[:] = np.random.choice(par.grid_n,size=par.simN) # Initial housing (discrete values)
        sim_path.a0[:] = par.mu_a0*np.random.lognormal(mean=1.3,sigma=par.sigma_a0,size=par.simN) # initial cash on hand

        # b. call simulation function
        with jit(self) as model:

            par = self.par
            sol_path = self.sol_path
            sim_path = self.sim_path

            simulate.lifecycle(sim_path,sol_path,par,path=True)