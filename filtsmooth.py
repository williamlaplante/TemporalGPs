
import numpy as np
from scipy.linalg import norm

class KalmanFilter():
    """
    Setup is as follows:

    X_k = A X_{k-1} + q, q ~ N(0, Sigma)

    Y_k = H X_{k} + r, r ~ N(0, R)

    with X_0 ~ N(0, Sig0)

    Hence, constant A, Sigma, H, R, and everything is linear and gaussian.
    
    """
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
        self.W = lambda y, y_hat : 1 / np.sqrt((1 + norm(y - y_hat)**2 / self.c_W**2))
        self.logW2 = lambda y, y_hat : np.log(self.W(y, y_hat)**2)
        self.logW2_jac = lambda y, y_hat : -(1 / (1 + norm(y - y_hat)**2 / self.c_W**2)) * (2/self.c_W**2) * norm(y - y_hat)

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


