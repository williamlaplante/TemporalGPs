
import numpy as np
from scipy.linalg import norm
from latent_process import LTISDE
from typing import Callable
from tqdm import tqdm
from minimax_tilting_sampler import TruncatedMVN


class QuadFormKalmanFilter():
    """
    Kalman Filter and Smoother where inputs are specified in terms of parameters of quadratic form truncated normal likelihood

    prior distribution (predict step): exp(-1/2 (X_k - m_prior)^T P_prior^{-1} (X_k - m_prior)), where
    
    m_prior = A @ X_{k-1}
    P_prior = A @ P_{k-1} @ A.T + Sigma

    likelihood, quadratic form (update step): exp(-1/2 ( X_k^T M X_k - 2 X_k^T v(y_k) ) ), where
    
    M is a matrix and v is a vector that need to be specified.

    For example, if 

    Y_k = H X_{k} + r, r ~ N(0, R)

    then

    M = H^T @ R^{-1} @ H
    v = H^T @ R^{-1} @ y_k

    The update step is thus:

    P_posterior = (P_prior^{-1} + self.M)^{-1}

    m_posterior = P_posterior @ (P_prior^{-1} @ m_prior + v(y))


    """
    def __init__(self, obs_grid : np.ndarray, obs : np.ndarray, subdiv : int, latent_process : LTISDE, M : np.ndarray, v : Callable, truncate=False, lb=-np.inf, ub=np.inf):

        #time stuff
        self.dt = (obs_grid[1] - obs_grid[0])/subdiv #Assume constant time step
        self.filter_grid = np.arange(obs_grid[0] - self.dt, obs_grid[-1] + self.dt, self.dt) #we subdivide the filtering grid, and start at tmin - dt (step 0 has no data)
        self.obs_grid = obs_grid

        #latent process stuff
        self.X_size = latent_process.X_size
        self.A = latent_process._A(self.dt)
        self.Sigma = latent_process._Sigma(self.dt)
        self.Sig0 = latent_process.Sig0
        self.m0 = np.zeros((self.X_size, 1))

        self.obs_size = 1 if len(obs.shape) == 1 else obs.shape[1]
        self.obs = obs.reshape(-1, self.obs_size)
        
        #observation process stuff
        y0 = self.obs[[0]]
        assert M.shape == (self.X_size, self.X_size), "M must be of size ({}, {}).".format(self.obs_size, self.X_size)
        assert v(y0).shape == (self.X_size, 1), "v must return an array of shape ({}, {})".format(self.X_size, 1)

        self.M = M
        self.v = v

        #filtering and update step stuff
        self.T = len(self.filter_grid)
        self.data_idx_offset = 0 # for update step
        self.data_idx = 0

        #truncation stuff
        self.truncate = truncate
        self.ub = ub
        self.lb = lb
        self.n_samples = int(1e4)

        return
    
    def truncation(self, mu, cov, idx):
    
        tmvn = TruncatedMVN(mu=mu, cov=cov, lb=self.lb[idx], ub=self.ub[idx])
        samples = tmvn.sample(self.n_samples)
        mu = samples.mean(axis=1)
        cov = np.cov(samples)
        return mu, cov
    

    def filtsmooth(self, filter_only = False):
        
        m = np.zeros((self.T, self.X_size, 1)) #mean
        P = np.zeros((self.T, self.X_size, self.X_size)) #covariance

        m[0] = self.m0
        P[0] = self.Sig0

        self.data_idx = 0

        #Filtering
        for k in tqdm(range(0, self.T - 1), desc="Filtering"):
            #predict for k+1
            m_prior = self.A @ m[k] 
            P_prior = self.A @ P[k] @ self.A.T + self.Sigma 

            if np.isclose(self.filter_grid[k+1], self.obs_grid).any():
                #update k+1 if there is data at k+1
                y = self.obs[[self.data_idx]]
                P[k+1] = np.linalg.inv(np.linalg.inv(P_prior) + self.M) 
                m[k+1] = P[k+1] @ ( np.linalg.inv(P_prior) @ m_prior + self.v(y) )

                if self.truncate:
                    m_unadjusted = m[k+1].flatten()
                    cov_unadjusted = P[k+1]
                    mu_adjusted, cov_adjusted = self.truncation(mu=m_unadjusted, cov=cov_unadjusted, idx=self.data_idx)
                    m[k+1] = mu_adjusted.reshape(-1,1)
                    P[k+1] = cov_adjusted

                self.data_idx+=1
            
            else:
                #no update if no data at k+1
                P[k+1] = P_prior
                m[k+1] = m_prior

        if filter_only:
            return m, P
        
        m_filt = m.copy()
        P_filt = P.copy()

        #Smoother
        for k in tqdm(range(len(self.filter_grid) - 2, -1, -1), desc='Smoothing'):
            m_prior = self.A @ m[k]
            P_prior = self.A @ P[k] @ self.A.T + self.Sigma

            C = P[k] @ self.A.T @ np.linalg.inv(P_prior)

            m[k] = m[k] + C @ (m[k+1] - m_prior)
            P[k] = P[k] + C @ (P[k+1] - P_prior) @ C.T
            
        return m_filt, P_filt, m, P



class KalmanFilter():
    """
    Setup is as follows:

    X_k = A X_{k-1} + q, q ~ N(0, Sigma)

    Y_k = H X_{k} + r, r ~ N(0, R)

    with X_0 ~ N(0, Sig0)

    Hence, we have constant A, Sigma, H, R, and everything is linear and gaussian.
    
    """
    def __init__(self, obs_grid : np.ndarray, obs : np.ndarray, subdiv : int, latent_process : LTISDE, H : np.ndarray, R : np.ndarray):

        #time stuff
        self.dt = (obs_grid[1] - obs_grid[0])/subdiv #Assume constant time step
        self.filter_grid = np.arange(obs_grid[0] - self.dt, obs_grid[-1] + self.dt, self.dt) #we subdivide the filtering grid, and start at tmin - dt (step 0 has no data)
        self.obs_grid = obs_grid

        #latent process stuff
        self.X_size = latent_process.X_size
        self.A = latent_process._A(self.dt)
        self.Sigma = latent_process._Sigma(self.dt)
        self.Sig0 = latent_process.Sig0
        self.m0 = np.zeros((self.X_size, 1))

        self.obs_size = 1 if len(obs.shape) == 1 else obs.shape[1]
        self.obs = obs.reshape(-1, self.obs_size)
        
        #observation process stuff
        assert H.shape == (self.obs_size, self.X_size), "H must be of size ({}, {}).".format(self.obs_size, self.X_size)
        assert R.shape == (self.obs_size, self.obs_size), "R must be of size ({}, {}).".format(self.obs_size, self.obs_size)

        self.H = H
        self.R = R

        #filtering and update step stuff
        self.T = len(self.filter_grid)
        self.data_idx = 0

        return
    
    def filtsmooth(self, filter_only = False):
        
        m = np.zeros((self.T, self.X_size, 1)) #mean
        P = np.zeros((self.T, self.X_size, self.X_size)) #covariance

        m[0] = self.m0
        P[0] = self.Sig0

        self.data_idx = 0

        #Filtering
        for k in range(0, self.T - 1):
            #predict for k+1
            m_prior = self.A @ m[k] 
            P_prior = self.A @ P[k] @ self.A.T + self.Sigma 

            if np.isclose(self.filter_grid[k+1], self.obs_grid).any():
                #update k+1 if there is data at k+1
                P[k+1] = np.linalg.inv(np.linalg.inv(P_prior) + self.H.T @ np.linalg.inv(self.R) @ self.H) 
                m[k+1] = P[k+1] @ (np.linalg.inv(P_prior) @ m_prior + self.H.T @ np.linalg.inv(self.R) @ self.obs[[self.data_idx], :])

                self.data_idx+=1
            
            else:
                #no update if no data at k+1
                P[k+1] = P_prior
                m[k+1] = m_prior

        if filter_only:
            return m, P
        
        m_filt = m.copy()
        P_filt = P.copy()

        #Smoother
        for k in range(len(self.filter_grid) - 2, -1, -1):
            m_prior = self.A @ m[k]
            P_prior = self.A @ P[k] @ self.A.T + self.Sigma

            C = P[k] @ self.A.T @ np.linalg.inv(P_prior)

            m[k] = m[k] + C @ (m[k+1] - m_prior)
            P[k] = P[k] + C @ (P[k+1] - P_prior) @ C.T
            
        return m_filt, P_filt, m, P




class ScoreMatchingFilter():
    def __init__(self, obs : np.ndarray, A : np.ndarray, Sigma : np.ndarray, H : np.ndarray, obs_noise : np.ndarray, Sig0 : np.ndarray):

        self.X_size = A.shape[1] #number of columns in A is the size of the state
        self.output_dim = 1 if len(obs.shape) == 1 else obs.shape[1]
        self.obs = obs

        self.A = A
        self.Sigma = Sigma
        self.H = H
        self.obs_noise = obs_noise
        self.Sig0 = Sig0

        self.c_W = 1
        self.beta = 1
        self.W = lambda y, y_hat : self.beta / np.sqrt((1 + norm(y - y_hat)**2 / self.c_W**2))
        self.logW2 = lambda y, y_hat : np.log(self.W(y, y_hat)**2)
        self.logW2_jac = lambda y, y_hat : -2 * (y - y_hat) / (self.c_W**2 + (y - y_hat)**2)

        return
    

    def filtsmooth(self, filter_only = False):
        
        m = np.zeros((len(self.obs), self.X_size)) #mean
        P = np.zeros((len(self.obs), self.X_size, self.X_size)) #covariance

        m[0] = np.zeros(self.X_size)
        P[0] = self.Sig0

        #Filter
        for k in range(0, len(self.obs) - 1):
            #predict
            m_prior = self.A @ m[k] # m_{k+1} = exp(F*dt) m_k
            P_prior = self.A @ P[k] @ self.A.T + self.Sigma # P_{k+1} = F P F^T + Sigma
            y_hat = self.H @ m_prior

            #update
            w_k = self.W(self.obs[k+1], y_hat) #weights for robust Kalman Filter
            m_w = self.H @ m_prior + self.obs_noise**2 * self.logW2_jac(self.obs[k+1], y_hat)
            Residuals = self.obs[k+1] - m_w
            R = (1/2) * (self.obs_noise**(4) / w_k**(2)) * np.eye(self.output_dim)
            Innovation = self.H @ P_prior @ self.H.T + R # here covariance R for the robust KF is now w**2 * R
            KalmanGain = P_prior @ self.H.T @ np.linalg.inv(Innovation)

            m[k+1] = m_prior + (KalmanGain @ Residuals).flatten()
            P[k+1] = P_prior - KalmanGain @ Innovation @ KalmanGain.T

        if filter_only:
            return m, P
        
        m_filt = m.copy()
        P_filt = P.copy()

        #Smoother
        for k in range(len(self.obs) - 2, -1, -1):
            m_prior = self.A @ m[k]
            P_prior = self.A @ P[k] @ self.A.T + self.Sigma

            C = P[k] @ self.A.T @ np.linalg.inv(P_prior)

            m[k] = m[k] + C @ (m[k+1] - m_prior)
            P[k] = P[k] + C @ (P[k+1] - P_prior) @ C.T
            
        return m_filt, P_filt, m, P




"""

THIS CODE IS CORRECT AND FUNCTIONS WELL BUT THE KALMAN FILTER EQUATIONS (PREDICT AND UPDATE STEPS) ARE IN THE WRONG FORMAT, 
AND IT ISN'T ADAPTED VERY WELL TO CONTINUOUS LATENT PROCESSES (E.G. DOESN'T HAVE SUBDIVIDE FEATURE)

class KalmanFilter():
    
    Setup is as follows:

    X_k = A X_{k-1} + q, q ~ N(0, Sigma)

    Y_k = H X_{k} + r, r ~ N(0, R)

    with X_0 ~ N(0, Sig0)

    Hence, we have constant A, Sigma, H, R, and everything is linear and gaussian.
    
    
    def __init__(self, obs : np.ndarray, A : np.ndarray, Sigma : np.ndarray, H : np.ndarray, R : np.ndarray, Sig0 : np.ndarray, weight_fct = ""):

        self.X_size = A.shape[1] #number of columns in A is the size of the state
        self.output_dim = 1 if len(obs.shape) == 1 else obs.shape[1]
        self.obs = obs

        self.A = A
        self.Sigma = Sigma
        self.H = H
        self.R = R
        self.Sig0 = Sig0

        self.weight_fct = weight_fct
        self.c_W = 1

        return
    
    def W(self, y, y_hat):
        if self.weight_fct == "":
            return 1
        
        if self.weight_fct == "IMQ":
            return 1 / np.sqrt((1 + norm(y - y_hat)**2 / self.c_W**2))

        else:
            raise Exception("weight function is invalid.")
    
    def filtsmooth(self, filter_only = False):
        
        m = np.zeros((len(self.obs), self.X_size)) #mean
        P = np.zeros((len(self.obs), self.X_size, self.X_size)) #covariance

        m[0] = np.zeros(self.X_size)
        P[0] = self.Sig0

        #Filter
        for k in range(0, len(self.obs) - 1):
            #predict
            m_prior = self.A @ m[k] # m_{k+1} = exp(F*dt) m_k
            P_prior = self.A @ P[k] @ self.A.T + self.Sigma # P_{k+1} = F P F^T + Sigma
            y_hat = self.H @ m_prior

            #update
            w_k = self.W(self.obs[k+1], y_hat) #weights for robust Kalman Filter
            Residuals = self.obs[k+1] - self.H @ m_prior
            Innovation = self.H @ P_prior @ self.H.T + self.R * w_k**(-2) # here covariance R for the robust KF is now w**2 * R
            KalmanGain = P_prior @ self.H.T @ np.linalg.inv(Innovation)

            m[k+1] = m_prior + KalmanGain @ Residuals
            P[k+1] = P_prior - KalmanGain @ Innovation @ KalmanGain.T

        if filter_only:
            return m, P
        
        m_filt = m.copy()
        P_filt = P.copy()

        #Smoother
        for k in range(len(self.obs) - 2, -1, -1):
            m_prior = self.A @ m[k]
            P_prior = self.A @ P[k] @ self.A.T + self.Sigma

            C = P[k] @ self.A.T @ np.linalg.inv(P_prior)

            m[k] = m[k] + C @ (m[k+1] - m_prior)
            P[k] = P[k] + C @ (P[k+1] - P_prior) @ C.T
            
        return m_filt, P_filt, m, P

"""