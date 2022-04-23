# For debugging

import time
import numpy as np
import numba as nb
import matplotlib.pyplot as plt

# load the DurableConsumptionModel module
from HousingModel import HousingModelClass

# Set the number of threads in numba
nb.set_num_threads(4)

# Setup model
T = 7
model = HousingModelClass(name='example_negm',par={'solmethod':'negm','T':T,'do_print':True})

# Run model
# model.precompile_numba()
model.solve()
