import numpy as np
import matplotlib.pyplot as plt

class Random2DGaussian:
    def __init__(self):
        self.minx = 0
        self.maxx = 10
        self.miny = 0
        self.maxy = 10
        mix = (self.maxx-self.minx) * np.random.random_sample() + self.minx
        miy = (self.maxy-self.miny) * np.random.random_sample() + self.miny
        self.mean = [mix, miy]
        
        eigvalx = (np.random.random_sample()*(self.maxx - self.minx)/5)**2
        eigvaly = (np.random.random_sample()*(self.maxy - self.miny)/5)**2
        
        D = np.diag([eigvalx, eigvaly])
        phi = np.random.uniform(0, 2 * np.pi)
        R = np.array([
            [np.cos(phi), -np.sin(phi)], 
            [np.sin(phi), np.cos(phi)]
            ])
        self.sigma = R @ D @ R.T
        
    def get_sample(self, n:int) -> np.array:
        """
        Generate n samples from the 2D Gaussian distribution.

        Parameters:
        n (int): Number of samples to generate.

        Returns:
        np.array: An array of shape (n, 2) containing the generated samples.
        """
        return np.random.multivariate_normal(mean=self.mean, cov=self.sigma, size=n)