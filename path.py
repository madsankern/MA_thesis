import numpy as np

def gen_path_R(par):

    # Permanent drop
    par.path_R[:] = par.R - par.R_drop # change the drop
    
    # Add other types of paths
    # Permanent increase/decrease, AR process