import numpy as np
from scipy.special import gamma, binom
from scipy.linalg import expm, solve_continuous_lyapunov


class LatentProcess():
    """
    We define a latent process of the form

    X'(t) = F X(t) + L dW(t), where

    n : number of variables in the latent process (dimensionality)
    p : number of derivatives in the latent process

    where
    
    X(t) = (f1, f1', f1'', ..., f1^{m-1}, f2, f2', ..., f2^{m-1}, fn', ...., fn^{m-1}). 
    
    """
    def __init__(self, num_variables : int, num_derivatives : int, F : np.ndarray, L : np.ndarray, Q : np.ndarray):
        self.num_variables = num_variables
        self.num_derivatives = num_derivatives

        self.F = F
        self.L = L
        self.Q = Q

        self.Sig0 = solve_continuous_lyapunov(self.F, -self.L @ self.Q @ self.L.T)

        return
    
    def _A(self, dt : float) -> np.ndarray:
        return expm(self.F * dt)
    
    def _Sigma(self, dt : float) -> np.ndarray:
        A = self._A(dt)
        return self.Sig0 - A @ self.Sig0 @ A.T


class MaternProcess(LatentProcess):
    """
    The prior Matern GP is defined via the stochastic process X(t) as

    X'(t) = F X(t) + L dW(t), 

    where dW(t) is white noise. We set
    
    n : number of variables (multi-output GP)
    p : number of derivatives (order of the Matern process)

    Therefore, 
    X(t) = (f1, f1', f1'', ..., f1^{m-1}, f2, f2', ..., f2^{m-1}, fn', ...., fn^{m-1}). 

    This leads to the following form for the matrices F and L:
    
        |0 1 0 ...      0|
        |0 0 1 ...      0|
    F = |...          ...|
        |0 0 0          1|
        |-a0 ... -a_{m-1}|
        |    ...         |
        |-a0 ... -a_{m-1}|
    
    L = | 0 0 ... 0 |
        | 0 0 ... 0 |
        | 1 0 ... 0 |
        | 0 1 ... 0 |
        |     ...   |
        | 0 0 ... 1 |

    The process is solved and discretized by the equation

    X_k = A_k X_{k-1} + q, q ~ N(0, Sigma_k)

    where 

    A_k = exp(F (t_k - t_{k-1}) )
    Sigma_k = P - A_k P A_k
    
    And P solves F P + P^T F + L Q L^T = 0, for Q = q * Id, and q being the spectral density constant of the Matern process.
    
    """

    def __init__(self, n : int, p : int, magnitude : float, lengthscale : float):

        #sizes
        self.n = n #e.g., x1, x2, x3 has n=3
        self.p = p #e.g., x, x', x'' has p=2
        self.X_size = (self.p+1) * self.n #x1, x2, ..., xn, x1', ..., xn', ..., x1^p, ... xn^p

        #matern process constants
        self.magnitude = magnitude
        self.lengthscale = lengthscale
        self.nu = 1/2 + self.p
        self.lamb = np.sqrt(2 * self.nu) / self.lengthscale
        self.spectral_density = self.magnitude * 2**self.n * np.pi **(self.n/2) * gamma(self.nu + self.n/2) * (2 * self.nu)**(self.nu) / gamma(self.nu) / (self.lengthscale ** (2 * self.nu))


        F = np.block([[np.zeros((n * p, n)), np.eye(n * p)],
                      [np.zeros((n, n)), np.zeros((n, n * p))]])
        
        F[-n:, :] = np.array([ n * [- binom(p+1, i) * self.lamb**(p+1 - i)] for i in range(p+1)]).flatten() #broadcasting

        L = np.block([[np.zeros((self.n * self.p, self.n))], 
                           [np.eye(self.n)]])
        
        Q = self.spectral_density * np.eye(self.n) #noise is of the form (w1, ..., wn) for (x1^p, ..., xn^p) only 

        
        super().__init__(num_variables=n, num_derivatives=p, F=F, L=L, Q=Q)

        return