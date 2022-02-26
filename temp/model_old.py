# This is a .py file which uses the EconModel class to define the household model

# import packages
import numpy as np
from types import SimpleNamespace
# import user generated funcs and jeppes here

# Define model class
class model_class():

    # Initialize
   def __init__(self,name=None):
        """ defines default attributes """

        # Names
        self.par = SimpleNamespace() # Parameters of the model
        self.sol = SimpleNamespace() # Solution functions
        self.sim = SimpleNamespace() # Simulation functions

    #########
    # Setup #
    #########
    # Setup values of all parameters

    def setup(self):

        # Initialize
        par = self.par

        par.beta = 0.96
        par.sigma = 2.0
        par.alpha = 0.7
