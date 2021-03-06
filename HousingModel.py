"""HousingModel
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
from consav import markov

# local modules
import utility
import trans
import last_period
import post_decision
import keeper
import adjuster
import simulate
import path

class HousingModelClass(ModelClass):
    
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
                           'Npsi','Nxi','Nm','Np','Nn','Nx','Na','Npb','Nshocks',
                           'do_print','do_print_period','do_marg_u','do_simple_wq']

    def setup(self):
        """ set baseline parameters """

        par = self.par

        # a. Horizon
        par.T = 200 #80 # NOTE find out what this should be, Number of iterations to find stationary solution
        par.path_T = 200 # Length of model solve along the path
        par.sim_T = 200 # Length of stationary simulation to ensure convergence
        
        # b. Preferences
        par.beta = 0.965
        par.rho = 2.0
        par.alpha = 0.82
        par.d_ubar = 1.5

        # c. Prices and costs
        par.R = 1.03
        par.ph = 1.0 # House price, set to equilibrium
        par.deltaa = 0.1 # maintenence cost
        par.phi = 1.0 # downpayment fraction
        par.eta = .01

        # d. Path for aggregate states
        par.path_R = np.full(par.path_T + par.T, par.R) # for impulse response
        par.path_ph = np.full(par.path_T + par.T, par.ph) # House price sequence
        par.R_drop = 0.01 #0.005 # Drop in interest rates for shock

        # e. Markov process income
        par.theta = 0.9
        par.sigma_y = 0.1

        # f. Purchase price - Ensure eq. price is in the interval
        par.Npb = 100
        par.pb_max = 2.0
        par.pb_min = 0.7
        
        # g. Taxes
        par.tauc = 0.0 # Wealth tax
        par.taug = 0.0 # Gains tax

        # h. Grids
        par.Ny = 5 # update this
        par.Nn = 10
        par.n_min = .5 #1.0
        par.n_max = 2
        par.Nm = 100
        par.m_max = 16.0
        par.Nx = 100
        par.x_max = par.m_max + 2*par.ph*par.n_max 
        par.Na = 100
        par.a_max = par.m_max-2.0

        # i. Simulation parameters - these must be based on steady state
        par.sigma_p0 = 0.2
        par.mu_d0 = 0.8
        par.sigma_d0 = 0.2
        par.mu_a0 = 0.6
        par.sigma_a0 = 0.1
        par.simN = 100_000
        par.sim_seed = 218
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

        # a. Grids
        self.create_grids()

        # b. Steady state
        self.solve_prep()
        self.simulate_prep()
        
        # c. Transition path
        self.solve_path_prep()
        self.simulate_path_prep()
            
    def create_grids(self):
        """ construct grids for states and shocks """
        
        par = self.par
        sim = self.sim
        sim_path = self.sim_path

        par.do_marg_u = True

        # a. States
        # par.grid_y = nonlinspace(par.y_min,par.y_max,par.Ny,1.1)
        par.grid_n = nonlinspace(par.n_min,par.n_max,par.Nn-1,1.0) # set to minus 1 again
        par.grid_n = np.insert(par.grid_n,0,0) # Add a zero in the beginning of the array
        par.grid_m = nonlinspace(0,par.m_max,par.Nm,1.1)
        par.grid_x = nonlinspace(0,par.x_max,par.Nx,1.1)
        par.grid_pb = nonlinspace(par.pb_min,par.pb_max,par.Npb,1.0)

        # Markov approx for income
        par.grid_y,par.p_mat,par.pi,par.p_mat_cum,par.pi_cum = markov.log_rouwenhorst(par.theta,par.sigma_y,par.Ny)
        
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
        sim_path.rand = np.zeros(shape=(par.path_T,par.simN))

        sim.rand[:,:] = np.random.uniform(size=(par.sim_T,par.simN))
        sim_path.rand[:,:] = np.random.uniform(size=(par.path_T,par.simN))
        
        sim.rand0 = np.random.uniform(size=par.simN) # Initial y state
        sim_path.rand0 = sim.rand[-1,:] # use last period for path

        # g. Initial allocation of housing and cash on hand
        sim.d0 = np.zeros(par.simN)
        sim.a0 = np.zeros(par.simN)

        #sim.d0[:] = np.random.choice(par.grid_n,size=par.simN)
        sim.a0[:] = par.mu_a0*np.random.lognormal(mean=1.3,sigma=par.sigma_a0,size=par.simN)

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
    # Solve #
    #########

    def precompile_numba(self):
        """ solve the model with very coarse grids and simulate with very few persons"""

        par = self.par

        tic = time.time()

        # a. Define parameters
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
        fastpar['Npb'] = 3
        fastpar['simN'] = 2

        # b. Apply parameters
        for key,val in fastpar.items():
            prev = getattr(par,key)
            setattr(par,key,val)
            fastpar[key] = prev

        self.allocate()

        # c. Solve and simulate in ss
        self.solve(do_assert=False)
        self.simulate()

        # d. Solve and simulate along path
        self.solve_path()
        self.simulate_path()

        # e. Reiterate parameters
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

    def solve(self,do_assert=False):
        """ Solve the model
        
        Args:
            do_assert (bool,optional): Check if policy and inverse value functions are non-negative and not NaN.
        """
        for t in reversed(range(self.par.T)):
            
            self.par.t = t

            with jit(self) as model:

                # i. Extract attributes (?)
                par = model.par
                sol = model.sol
                
                # ii. Last period
                if t == par.T-1:
                    # tic = time.time()
                    # o. Solve last period
                    last_period.solve(t,sol,par,par.ph)
                    # toc = time.time()
                    # print(f'last_period computed in {toc-tic:.1f} secs')
                    # oo. Check solution for errors
                    if do_assert:
                        assert np.all((sol.c_keep[t] >= 0) & (np.isnan(sol.c_keep[t]) == False))
                        assert np.all((sol.inv_v_keep[t] >= 0) & (np.isnan(sol.inv_v_keep[t]) == False))
                        assert np.all((sol.d_adj[t] >= 0) & (np.isnan(sol.d_adj[t]) == False))
                        assert np.all((sol.c_adj[t] >= 0) & (np.isnan(sol.c_adj[t]) == False))
                        assert np.all((sol.inv_v_adj[t] >= 0) & (np.isnan(sol.inv_v_adj[t]) == False))

                # iii. All other periods
                else:
                    # tic = time.time()
                    # o. Compute post-decision functions
                    post_decision.compute_wq(t,par.R,sol,par,par.ph,compute_q=True)
                    # toc = time.time()
                    # print(f'post_decision computed in {toc-tic:.1f} secs')
                    if do_assert:
                        assert np.all((sol.inv_w[t] > 0) & (np.isnan(sol.inv_w[t]) == False)), t 
                        assert np.all((sol.q[t] > 0) & (np.isnan(sol.q[t]) == False)), t
                    # tic = time.time()    
                    # oo. Solve keeper problem
                    keeper.solve_keep(t,sol,par)
                    # toc = time.time()
                    # print(f'keeper computed in {toc-tic:.1f} secs')
                    if do_assert:
                        assert np.all((sol.c_keep[t] >= 0) & (np.isnan(sol.c_keep[t]) == False)), t
                        assert np.all((sol.inv_v_keep[t] >= 0) & (np.isnan(sol.inv_v_keep[t]) == False)), t
                    # tic = time.time()
                    # ooo. Solve adjuster problem
                    adjuster.solve_adj(t,sol,par,par.ph)                  
                    # toc = time.time()
                    # print(f'adjuster computed in {toc-tic:.1f} secs')
                    if do_assert:
                        assert np.all((sol.d_adj[t] >= 0) & (np.isnan(sol.d_adj[t]) == False)), t
                        assert np.all((sol.c_adj[t] >= 0) & (np.isnan(sol.c_adj[t]) == False)), t
                        assert np.all((sol.inv_v_adj[t] >= 0) & (np.isnan(sol.inv_v_adj[t]) == False)), t

                # iv. Compute distance to previous iteration
                if t < par.T-1:
                    dist1 = np.abs(np.max(sol.c_keep[t+1,:,:,:,:] - sol.c_keep[t,:,:,:,:]))
                    dist2 = np.abs(np.max(sol.c_adj[t+1,:,:,:] - sol.c_adj[t,:,:,:]))
                    sol.dist[t] = np.max(dist1,dist2)

        # b. Insert last iteration in all periods (infinite horizon)
        with jit(self) as model:

            par = self.par
            sol_path = self.sol_path
            sol = self.sol

            # i. Keeper
            sol.c_keep[:] = sol.c_keep[0]
            sol.inv_v_keep[:] = sol.inv_v_keep[0]
            sol.inv_marg_u_keep[:] = sol.inv_marg_u_keep[0]

            # ii. Adjuster
            sol.d_adj[:] = sol.d_adj[0]
            sol.c_adj[:] = sol.c_adj[0]
            sol.inv_v_adj[:] = sol.inv_v_adj[0]
            sol.inv_marg_u_adj[:] = sol.inv_marg_u_adj[0]
                
            # iii. Post decision
            sol.inv_w[:] = sol.inv_w[0]
            sol.q[:] = sol.q[0]
            sol.q_c[:] = sol.q_c[0]
            sol.q_m[:] = sol.q[0]        

    ############
    # Simulate #
    ############

    def simulate_prep(self):
        """ allocate memory for simulation """

        par = self.par
        sim = self.sim

        # a. Household utility
        sim.utility = np.zeros(par.simN)

        # b. States and choices
        sim_shape = (par.sim_T,par.simN)
        sim.y = np.zeros(sim_shape)
        sim.m = np.zeros(sim_shape)
        sim.n = np.zeros(sim_shape)
        sim.discrete = np.zeros(sim_shape,dtype=np.int)
        sim.d = np.zeros(sim_shape)
        sim.c = np.zeros(sim_shape)
        sim.a = np.zeros(sim_shape)
        sim.pb = np.zeros(sim_shape)
        
        # c. Euler errors
        euler_shape = (par.sim_T-1,par.simN)
        sim.euler_error = np.zeros(euler_shape)
        sim.euler_error_c = np.zeros(euler_shape)
        sim.euler_error_rel = np.zeros(euler_shape)

        # d. Income states
        sim.state = np.zeros((par.sim_T,par.simN),dtype=np.int_)
        sim.state_lag = np.zeros(par.simN) # not used

    def simulate(self,do_utility=False,do_euler_error=False):
        """ simulate the model """

        par = self.par
        sol = self.sol
        sim = self.sim

        # a. Ensure that paths for R and p are constantly equal to their ss value
        par.path_R[:] = par.R
        par.path_ph[:] = par.ph

        # b. Call
        with jit(self) as model:

            par = model.par
            sol = model.sol
            sim = model.sim
            sim_path = model.sim_path # Can be removed?

            simulate.monte_carlo(sim,sol,par,path=False)

    #########################
    # Solve Transition Path #
    #########################

    def solve_path_prep(self):
        """ allocate memory for solution along a transition path"""

        par = self.par
        sol_path = self.sol_path

        # NOTE the length of the time horizon
        
        # a. Keeper
        keep_shape_path = (par.path_T+par.T,par.Npb,par.Ny,par.Nn,par.Nm)
        sol_path.c_keep = np.zeros(keep_shape_path)
        sol_path.inv_v_keep = np.zeros(keep_shape_path)
        sol_path.inv_marg_u_keep = np.zeros(keep_shape_path)

        # b. Adjuster
        adj_shape_path = (par.path_T+par.T,par.Npb,par.Ny,par.Nx)
        sol_path.d_adj = np.zeros(adj_shape_path)
        sol_path.c_adj = np.zeros(adj_shape_path)
        sol_path.inv_v_adj = np.zeros(adj_shape_path)
        sol_path.inv_marg_u_adj = np.zeros(adj_shape_path)

        # c. Post decision
        post_shape_path = (par.path_T+par.T-1,par.Npb,par.Ny,par.Nn,par.Na)
        sol_path.inv_w = np.nan*np.zeros(post_shape_path)
        sol_path.q = np.nan*np.zeros(post_shape_path)
        sol_path.q_c = np.nan*np.zeros(post_shape_path)
        sol_path.q_m = np.nan*np.zeros(post_shape_path)

    def solve_path(self): # Add options for type of shock
        '''Solve the household problem along a transition path'''

        # a. Generate exogenous path of interest rates
        path.gen_path_R(self.par)

        for t in reversed(range(self.par.path_T + self.par.T)):

            with jit(self) as model:

                par = self.par
                sol_path = self.sol_path

                # i. Update interest rates and house prices from path
                R = par.path_R[t]
                ph = par.path_ph[t]
                
                # ii. Last period
                if t == (par.path_T+par.T)-1:

                    last_period.solve(t,sol_path,par,ph)

                else:
                    
                    # o. Compute post decision value
                    post_decision.compute_wq(t,R,sol_path,par,ph,compute_q=True)

                    # oo. Solve keeper
                    keeper.solve_keep(t,sol_path,par)

                    # ooo. Solve adjuster
                    adjuster.solve_adj(t,sol_path,par,ph)

        # b. Replace end points with the 50'th iteration to remove terminal period effect
        with jit(self) as model:

            par = self.par
            sol_path = self.sol_path

            # i. Keeper
            sol_path.c_keep[par.path_T:] = sol_path.c_keep[par.path_T]
            sol_path.inv_v_keep[par.path_T:] = sol_path.inv_v_keep[par.path_T]
            sol_path.inv_marg_u_keep[par.path_T:] = sol_path.inv_marg_u_keep[par.path_T]

            # ii. Adjuster
            sol_path.d_adj[par.path_T:] = sol_path.d_adj[par.path_T]
            sol_path.c_adj[par.path_T:] = sol_path.c_adj[par.path_T]
            sol_path.inv_v_adj[par.path_T:] = sol_path.inv_v_adj[par.path_T]
            sol_path.inv_marg_u_adj[par.path_T:] = sol_path.inv_marg_u_adj[par.path_T]
                
            # iii. Post decision
            sol_path.inv_w[par.path_T:] = sol_path.inv_w[par.path_T]
            sol_path.q[par.path_T:] = sol_path.q[par.path_T]
            sol_path.q_c[par.path_T:] = sol_path.q_c[par.path_T]
            sol_path.q_m[par.path_T:] = sol_path.q[par.path_T]       

    ############################
    # Simulate Transition Path #
    ############################
    
    def simulate_path_prep(self):
        """ allocate memory for simulation """

        par = self.par
        sim_path = self.sim_path

        # a. initial allocation
        sim_path.d0 = np.zeros(par.simN)
        sim_path.a0 = np.zeros(par.simN)
        sim_path.state_lag = np.zeros(par.simN)

        # b. Household utility
        sim_path.utility = np.zeros(par.simN)

        # c. States and choices
        sim_shape_path = (par.path_T,par.simN) # NOTE the horizon
        sim_path.y = np.zeros(sim_shape_path)
        sim_path.m = np.zeros(sim_shape_path)
        sim_path.n = np.zeros(sim_shape_path)
        sim_path.discrete = np.zeros(sim_shape_path,dtype=np.int)
        sim_path.d = np.zeros(sim_shape_path)
        sim_path.c = np.zeros(sim_shape_path)
        sim_path.a = np.zeros(sim_shape_path)
        sim_path.pb = np.zeros(sim_shape_path)

        # d. Income states
        sim_path.state = np.zeros((par.path_T,par.simN),dtype=np.int_)

    def simulate_path(self):
        '''Simulate a panel of households along a transition path'''
    
        # par = self.par
        # sol_path = self.sol_path
        sim_path = self.sim_path
        sim = self.sim # ss simulation

        # a. Last period of ss simulation as initial allocation
        sim_path.d0[:] = sim.d[-1,:]
        sim_path.a0[:] = sim.a[-1,:]
        sim_path.state_lag[:] = sim.state[-1,:]

        # b. call simulation function
        with jit(self) as model:

            par = self.par
            sol_path = self.sol_path
            sim_path = self.sim_path
            sim = self.sim

            simulate.monte_carlo(sim_path,sol_path,par,path=True)