# This is a .py file which uses the EconModel class to define the household model

# import packages
import numpy as np
from types import SimpleNamespace
from EconModel import EconModelClass, jit
from consav.grids import nonlinspace

# Import user generated packages here
# import last_period etc.

# Define model class
class model_class():

    # Initialize
    def __init__(self,name=None):
        """ defines default attributes """

        # Names
        self.par = SimpleNamespace() # Parameters of the model
        self.sol = SimpleNamespace() # Solution functions
        self.sim = SimpleNamespace() # Simulation functions

    def settings(self): # Think about what 'self' really means
        pass

    #########
    # Setup #
    #########
    # Setup values of all parameters

    def setup(self):

        # Initialize
        par = self.par

        # Preferences
        par.beta = 0.965 # discount factor
        par.rho = 2.0 # elasticity of substitution 
        par.alpha = 0.7 # weight on consumption

        # Returns and income
        par.R = 1.03 # interest rate
        par.delta = 0.01 # Maintenance cost
        par.p = 2.0 # House price
        # Parameters for the markov income process

        # Horizon - this is removed later
        par.T

        # Grid settings
        par.Nh = 10 # Number of house sizes
        par.h_max = 10.0 # Largets house
        par.h_min = 0.0
        
        par.Nm = 100
        par.m_max = 10.0    
        
        par.Nx = 100
        par.x_max = par.m_max + par.h_max # Adjust this with house price at some point
        
        par.Na = 100
        par.a_max = par.m_max + 1.0

    def allocate(self):
        """ allocate model, i.e. create grids and allocate solution and simluation arrays """

        self.create_grids()
    
    def create_grids(self):
        """ condstruct grids for states """

        par = self.par

        # Housing - this must be a linear grid
        par.grid_h = np.linspace(par.h_min, par.h_max, par.Nh)

        # States
        par.grid_m = nonlinspace(0, par.m_max, par.Nm, 1.1)
        par.grid_x = nonlinspace(0, par.x_max, par.Nx, 1.1)

        # Post decision state
        par.grid_a = nonlinspace(0,par.a_max, par.Na, 1.1)

    #########
    # Solve #
    #########

    def solve_prep(self):
        """ allocate memory for solution """

        par = self.par
        sol = self.sol # I think this must definered somewhere in the beginning

        # Initialize
        shape = (par.Nh, par.Nm) # Adjust this later with add. states
        
        sol.c_keep = np.zeros(shape)
        sol.inv_v_keep  = np.zeros(shape)
        sol.inv_marg_u_keep = np.zeros(shape)

        sol.h_adj = np.zeros(shape)
        sol.c_adj = np.zeros(adj_shape)
        sol.inv_v_adj = np.zeros(adj_shape)
        sol.inv_marg_u_adj = np.zeros(adj_shape)

        # Add shape for post decision state

    def solve(self):
        """ solve the model """

        # Solve with backwards induction
        # For now with a deterministic horizon
        for t in reversed(range(self.par.T)):

            # Something from the EconModel class
            with jit(self) as model:

                par = model.par
                sol = model.sol

                # Solve last period
                if t == par.T-1:

                    last_period.solve(t,sol,par)

                # All other periods
                else:

                    # compute post decision function
                    # post_decision.compute_wq etc.

                    # Solve keeper problem (Upper envelope)

                    # Solve adjuster (code in NVFI file)
                    # nvfi.solve_adj(input)

