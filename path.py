import numpy as np

def gen_path_R(par):

    # Permanent drop
    par.path_R[par.sim_T:] = par.R - 0.002 # change the drop
    
    # Add other types of paths
    # Permanent increase/decrease, AR process