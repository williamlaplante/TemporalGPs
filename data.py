import numpy as np
from typing import Callable

class Dataset():
    def __init__(self, tmin : float, tmax : float, num_data_pts : int, func : Callable, dist : str, params : dict):
        self.tmin = tmin
        self.tmax = tmax

        self.num_data_pts = num_data_pts

        self.obs_grid = np.linspace(tmin, tmax, num_data_pts)
        self.dt = float(self.obs_grid[1] - self.obs_grid[0])

        self.f = func
        self.dist = dist
        self.params = params

        return
    
    def noise(self):
        if self.dist == 'normal':
            return np.random.normal(**self.params, size=self.num_data_pts)
        
        elif self.dist == 'beta':
            return np.random.beta(**self.params, size=self.num_data_pts)
        
        elif self.dist == 'lognormal':
            return np.random.lognormal(**self.params, size=self.num_data_pts)
        
        elif self.dist == 'exponential':
            return np.random.exponential(**self.params, size=self.num_data_pts)
        
        elif self.dist == 'weibull':
            return np.random.weibull(**self.params, size=self.num_data_pts)
        
        else:
            raise Exception("Distribution not available.")

    def observations(self):
        obs = self.f(self.obs_grid) + self.noise()

        return self.obs_grid, obs




"""
KEEPING THIS TO REMEMBER HOW TO DO OUTLIERS

def create_periodic_data(func, tmin, tmax, num_data_pts, scale, outliers=False):

    grid = np.linspace(tmin, tmax, num_data_pts)
    dt = float(grid[1]-grid[0])
    
    obs = func(2 * np.pi * grid) + np.random.normal(loc=0, scale=scale, size=len(grid)) 

    #Introducing outliers
    if outliers:
        num_outliers = 10
        start_idx = 5
        idx = np.random.randint(start_idx, num_data_pts, size=num_outliers)
        outliers_vals = np.sin((2 * np.pi * grid[idx])) + np.random.normal(loc=0, scale=10*scale, size=num_outliers)
        obs[idx] = outliers_vals

    return (grid, obs)"""