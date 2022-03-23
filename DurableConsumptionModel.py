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
import figs

class DurableConsumptionModelClass(ModelClass): # Rename
    
    #########
    # Setup #
    #########
    
    def settings(self):
        """ choose settings """

        # a. namespaces
        self.namespaces = []
        
        # b. other attributes
        self.other_attrs = []
        
        # c. savefolder
        self.savefolder = 'saved'

        # d. list not-floats for safe type inference
        self.not_floats = ['solmethod','T','t','simN','sim_seed','cppthreads',
                           'Npsi','Nxi','Nm','Np','Nn','Nx','Na','Nshocks',
                           'do_2d','do_print','do_print_period','do_marg_u','do_simple_wq']

    def setup(self):
        """ set baseline parameters """

        par = self.par

        # a. baseline parameters
        par.do_2d = False

        # horizon
        par.T = 100
        
        # preferences
        par.beta = 0.965
        par.rho = 2.0
        par.alpha = 0.9
        par.d_ubar = 0.1

        # returns and income
        par.R = 1.03
        par.tau = 0.10
        par.deltaa = 0.15
        par.pi = 0.0 # what is this
        par.mu = 0.5 # what is this
        par.ph = 2.0 # House price - rename to p

        # Markov process stuff
        par.p_12 = 0.33
        par.p_21 = 0.33
        par.p_mat = np.array([ 
            [1-par.p_12, par.p_12], 
            [par.p_21, 1-par.p_21]])
        par.p_mat_cum = np.array([np.cumsum(par.p_mat[i,:]) for i in range(2)])

        par.pi = np.array([1/2,1/2]) # stationary distribution
        par.pi_cum = np.array(np.cumsum(par.pi))

        # p_buy
        par.Npb = 2 # points in the grid
        par.pb_max = 5.0 # max value
        par.pb_min = 0.1 # min value
        
        # taxes
        par.tauc = 0.0 # Wealth tax
        par.taug = 0.0 # Gains tax

        # grids - rename p to y
        par.Np = 2 #6 # update this
        par.p_min = 0.5
        par.p_max = 2.0
        par.Nn = 10 # For now set 5 levels of housing
        par.n_max = 1.0
        par.Nm = 100
        par.m_max = 10.0    
        par.Nx = 100
        par.x_max = par.m_max + par.n_max
        par.Na = 100
        par.a_max = par.m_max+1.0

        # simulation - these must be based on steady state
        par.sigma_p0 = 0.2
        par.mu_d0 = 0.8
        par.sigma_d0 = 0.2
        par.mu_a0 = 0.2
        par.sigma_a0 = 0.1
        par.simN = 1000
        par.sim_seed = 1998
        par.euler_cutoff = 0.02

        # misc
        par.solmethod = 'negm' # this might also be redundant
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

        par.do_marg_u = True

        # a. states
        par.grid_p = nonlinspace(par.p_min,par.p_max,par.Np,1.1)
        par.grid_n = nonlinspace(0,par.n_max,par.Nn,1.1) # Grid over housing, can be nonlinspace
        par.grid_m = nonlinspace(0,par.m_max,par.Nm,1.1)
        par.grid_x = nonlinspace(0,par.x_max,par.Nx,1.1)
        par.grid_pb = nonlinspace(par.pb_min,par.pb_max,par.Npb,1.1) # Grid for house purchase price
        
        # b. post-decision states
        par.grid_a = nonlinspace(0,par.a_max,par.Na,1.1)

        # d. set seed
        np.random.seed(par.sim_seed)

        # e. timing
        par.time_w = np.zeros(par.T)
        par.time_keep = np.zeros(par.T)
        par.time_adj = np.zeros(par.T)
        par.time_adj_full = np.zeros(par.T)

    def checksum(self,simple=False,T=1):
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

        # a. define
        fastpar = dict()
        fastpar['do_print'] = False
        fastpar['do_print_period'] = False
        fastpar['T'] = 2
        fastpar['Np'] = 2
        fastpar['Nn'] = 3
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
        # self.simulate()  # uncomment this when the model runs

        # e. reiterate
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

        # a. standard
        keep_shape = (par.T,par.Npb,par.Np,par.Nn,par.Nm)
        
        sol.c_keep = np.zeros(keep_shape)
        sol.inv_v_keep = np.zeros(keep_shape)
        sol.inv_marg_u_keep = np.zeros(keep_shape)

        adj_shape = (par.T,par.Npb,par.Np,par.Nx)
        sol.d_adj = np.zeros(adj_shape)
        sol.c_adj = np.zeros(adj_shape)
        sol.inv_v_adj = np.zeros(adj_shape)
        sol.inv_marg_u_adj = np.zeros(adj_shape)
            
        post_shape = (par.T-1,par.Npb,par.Np,par.Nn,par.Na)
        sol.inv_w = np.nan*np.zeros(post_shape)
        sol.q = np.nan*np.zeros(post_shape)
        sol.q_c = np.nan*np.zeros(post_shape)
        sol.q_m = np.nan*np.zeros(post_shape)
        # print(post_shape)

    def solve(self,do_assert=True):
        """ solve the model
        
        Args:

            do_assert (bool,optional): make assertions on the solution
        
        """

        tic = time.time()
            
        # backwards induction
        for t in reversed(range(self.par.T)):
            
            self.par.t = t

            with jit(self) as model:

                par = model.par
                sol = model.sol
                
                # i. last period
                if t == par.T-1:

                    last_period.solve(t,sol,par)

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
                    
                    post_decision.compute_wq(t,sol,par,compute_q=True)

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
                    
                    nvfi.solve_adj(t,sol,par)                  

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

    ############
    # simulate #
    ############

    def simulate_prep(self):
        """ allocate memory for simulation """

        par = self.par
        sim = self.sim

        # a. initial and final
        sim.p0 = np.zeros(par.simN)

        # else:
        sim.d0 = np.zeros(par.simN)
        sim.d10 = np.zeros(0) # not used
        sim.d20 = np.zeros(0) # not used 
        sim.a0 = np.zeros(par.simN)

        sim.utility = np.zeros(par.simN)

        # b. states and choices
        sim_shape = (par.T,par.simN)
        sim.p = np.zeros(sim_shape)
        sim.m = np.zeros(sim_shape)
        sim.n = np.zeros(sim_shape)
        sim.n1 = np.zeros((0,0)) # not used
        sim.n2 = np.zeros((0,0)) # not used 
        sim.discrete = np.zeros(sim_shape,dtype=np.int)
        sim.d = np.zeros(sim_shape)
        sim.d1 = np.zeros((0,0)) # not used
        sim.d2 = np.zeros((0,0)) # not used   
        sim.c = np.zeros(sim_shape)
        sim.a = np.zeros(sim_shape)
        
        # c. euler
        euler_shape = (par.T-1,par.simN)
        sim.euler_error = np.zeros(euler_shape)
        sim.euler_error_c = np.zeros(euler_shape)
        sim.euler_error_rel = np.zeros(euler_shape)

        # d. shocks - I only need shocks to income (one variable)
        sim.psi = np.zeros((par.T,par.simN))
        sim.xi = np.zeros((par.T,par.simN))
        sim.rand = np.zeros((par.T,par.simN))

        sim.state = np.zeros((par.T,par.simN),dtype=np.int_) # Container for income states


    def simulate(self,do_utility=False,do_euler_error=False):
        """ simulate the model """

        par = self.par
        sol = self.sol
        sim = self.sim

        tic = time.time()

        # a. random shocks
        sim.p0[:] = np.random.lognormal(mean=0,sigma=par.sigma_p0,size=par.simN) # Initial permanent income, can be removed
        sim.d0[:] = np.random.choice(par.grid_n,size=par.simN) # Initial housing
        sim.a0[:] = par.mu_a0*np.random.lognormal(mean=0,sigma=par.sigma_a0,size=par.simN) # initial cash on hand

        # Set shocks in each period
        sim.rand[:,:] = np.random.uniform(size=(par.T,par.simN)) # move to model.py as numba does not like this

        # b. call
        with jit(self) as model:

            par = model.par
            sol = model.sol
            sim = model.sim
            # print(sim.rand)
            simulate.lifecycle(sim,sol,par)

        toc = time.time()
        
        if par.do_print:
            print(f'model simulated in {toc-tic:.1f} secs')

        # # d. euler errors
        # def norm_euler_errors(model):
        #     return np.log10(abs(model.sim.euler_error/model.sim.euler_error_c)+1e-8)

        # tic = time.time()        
        # if do_euler_error:

        #     with jit(self) as model:

        #         par = model.par
        #         sol = model.sol
        #         sim = model.sim

        #         simulate.euler_errors(sim,sol,par)
            
        #     sim.euler_error_rel[:] = norm_euler_errors(self)
        
        # toc = time.time()
        # if par.do_print:
        #     print(f'euler errors calculated in {toc-tic:.1f} secs')

        # # e. utility
        # tic = time.time()        
        # if do_utility:
        #     simulate.calc_utility(sim,sol,par)
        
        # toc = time.time()
        # if par.do_print:
        #     print(f'utility calculated in {toc-tic:.1f} secs')

    ########
    # figs #
    ########

    def decision_functions(self):
        figs.decision_functions(self)

    def egm(self):        
        figs.egm(self)

    def lifecycle(self):        
        figs.lifecycle(self)

    ###########
    # analyze #
    ###########

    # def analyze(self,solve=True,do_assert=True,**kwargs):
        
    #     par = self.par

    #     for key,val in kwargs.items():
    #         setattr(par,key,val)
        
    #     self.create_grids()

    #     # solve and simulate
    #     if solve:
    #         self.precompile_numba()
    #         self.solve(do_assert)
    #     self.simulate(do_euler_error=True,do_utility=True)

    #     # print
    #     self.print_analysis()

    # def print_analysis(self):

    #     par = self.par
    #     sim = self.sim

    #     def avg_euler_error(model,I):
    #         if I.any():
    #             return np.nanmean(model.sim.euler_error_rel.ravel()[I])
    #         else:
    #             return np.nan

    #     def percentile_euler_error(model,I,p):
    #         if I.any():
    #             return np.nanpercentile(model.sim.euler_error_rel.ravel()[I],p)
    #         else:
    #             return np.nan

    #     # population
    #     keepers = sim.discrete[:-1,:].ravel() == 0
    #     adjusters = sim.discrete[:-1,:].ravel() > 0
    #     adjusters_both = sim.discrete[:-1,:].ravel() == 1
    #     adjusters_d1 = sim.discrete[:-1,:].ravel() == 2
    #     adjusters_d2 = sim.discrete[:-1,:].ravel() == 3
    #     everybody = keepers | adjusters

    #     # print
    #     time = par.time_w+par.time_adj+par.time_keep
    #     txt = f'Name: {self.name} (solmethod = {par.solmethod})\n'
    #     txt += f'Grids: (p,n,m,x,a) = ({par.Np},{par.Nn},{par.Nm},{par.Nx},{par.Na})\n'
        
    #     txt += 'Timings:\n'
    #     txt += f' total: {np.sum(time):.1f}\n'
    #     txt += f'     w: {np.sum(par.time_w):.1f}\n'
    #     txt += f'  keep: {np.sum(par.time_keep):.1f}\n'
    #     txt += f'   adj: {np.sum(par.time_adj):.1f}\n'
    #     txt += f'Utility: {np.mean(sim.utility):.6f}\n'
        
    #     txt += 'Euler errors:\n'
    #     txt += f'     total: {avg_euler_error(self,everybody):.2f} ({percentile_euler_error(self,everybody,5):.2f},{percentile_euler_error(self,everybody,95):.2f})\n'
    #     txt += f'   keepers: {avg_euler_error(self,keepers):.2f} ({percentile_euler_error(self,keepers,5):.2f},{percentile_euler_error(self,keepers,95):.2f})\n'
    #     txt += f' adjusters: {avg_euler_error(self,adjusters):.2f} ({percentile_euler_error(self,adjusters,5):.2f},{percentile_euler_error(self,adjusters,95):.2f})\n'
        
    #     if par.do_2d:
    #         txt += f'      both: {avg_euler_error(self,adjusters_both):.2f} ({percentile_euler_error(self,adjusters_both,10):.2f},{percentile_euler_error(self,adjusters_both,90):.2f})\n'
    #         txt += f'        d1: {avg_euler_error(self,adjusters_d1):.2f} ({percentile_euler_error(self,adjusters_d1,10):.2f},{percentile_euler_error(self,adjusters_d1,90):.2f})\n'
    #         txt += f'        d2: {avg_euler_error(self,adjusters_d2):.2f} ({percentile_euler_error(self,adjusters_d2,10):.2f},{percentile_euler_error(self,adjusters_d2,90):.2f})\n'

    #     txt += 'Moments:\n'
    #     if par.do_2d:
    #         txt += f' adjuster share: {np.mean(sim.discrete > 0):.3f}\n'
    #         txt += f'           both: {np.mean(sim.discrete == 1):.3f}\n'
    #         txt += f'        only d1: {np.mean(sim.discrete == 2):.3f}\n'
    #         txt += f'        only d2: {np.mean(sim.discrete == 3):.3f}\n'
    #     else:
    #         txt += f' adjuster share: {np.mean(sim.discrete):.3f}\n'
        
    #     txt += f'         mean c: {np.mean(sim.c):.3f}\n'
    #     txt += f'          var c: {np.var(sim.c):.3f}\n'

    #     if par.do_2d:
    #         txt += f'         mean d1: {np.mean(sim.d1):.3f}\n'
    #         txt += f'          var d1: {np.var(sim.d1):.3f}\n'
    #         txt += f'         mean d2: {np.mean(sim.d2):.3f}\n'
    #         txt += f'          var d2: {np.var(sim.d2):.3f}\n'
    #     else:
    #         txt += f'         mean d: {np.mean(sim.d):.3f}\n'
    #         txt += f'          var d: {np.var(sim.d):.3f}\n'

    #     print(txt)