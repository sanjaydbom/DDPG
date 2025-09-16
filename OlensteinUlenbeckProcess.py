import numpy as np

#dx_t = -theta * x_tdt + sigma * dW_t

class OUP():
    def __init__(self, theta = 0.15, sigma = 0.2):
        self.x_t = 0
        self.theta = -1 * theta
        self.sigma = sigma

    def __call__(self):
        dW_t = np.random.randn()
        dx_t = self.theta * self.x_t + self.sigma * dW_t
        self.x_t += dx_t
        return self.x_t
    
    def reset(self):
        self.x_t = 0
