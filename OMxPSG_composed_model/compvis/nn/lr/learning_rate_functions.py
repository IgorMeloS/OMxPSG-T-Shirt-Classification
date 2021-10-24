# =============================================================================
# Learn rate scheduler functions
# =============================================================================
import numpy as np
class LRFunc:
    """Class with learn rate scheculer functions"""
    def step_decay(epoch, lr):
    
        init_alpha = 0.001 # initial learning rate
        factor = 0.99
        step = 5 # step size
        if epoch  == 0:
            lr = init_alpha
            print("The intitial learning rate is {:f}".format(lr))
        elif   epoch % step  == 0:
            tol = 1e-6
            lr = init_alpha * (factor ** np.floor((1 + epoch) / step)) + tol
            print("Learning rate updated {: f}".format(lr))
        else:
            lr = lr
        return lr