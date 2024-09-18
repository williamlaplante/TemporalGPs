import numpy as np
from scipy.special import gamma, binom
from scipy.linalg import expm, solve_continuous_lyapunov


class LTISDE():
    """
    We define a linear time invariant process of the form

    X'(t) = F X(t) + L dW(t), where

    w(t) : white noise process
    F, L : constant matrix defining the process
    Q : spectral density matrix
    n : number of variables in the latent process (dimensionality)
    p : number of derivatives in the latent process

    and X(t) must be of the form
    
    X(t) = (f1, f1', f1'', ..., f1^{m-1}, f2, f2', ..., f2^{m-1}, fn', ...., fn^{m-1}). 
    
    The process is solved and discretized by the equation

    X_k = A_k X_{k-1} + q, q ~ N(0, Sigma_k)

    where 

    A_k = exp(F (t_k - t_{k-1}) )
    Sigma_k = P - A_k P A_k
    
    And P solves F P + P^T F + L Q L^T = 0, for Q = q * Id, and q being the spectral density constant of the Matern process.
    
    """
    def __init__(self, n : int, p : int, F : np.ndarray, L : np.ndarray, Q : np.ndarray):
        self.n = n
        self.p = p
        self.X_size = (self.p+1) * self.n

        assert F.shape == (n * (p+1), n * (p+1)), "F's shape is supposed to be n * (p+1) by n * (p+1)"
        assert L.shape == (n * (p+1), n), "L's shape is supposed to be n * (p+1) by n"
        assert Q.shape == (n, n), "Q's shape is supposed to be n by n"

        self.F = F
        self.L = L
        self.Q = Q 

        assert (np.linalg.eig(self.F).eigenvalues < 0).all(), "Matrix F must be Hurwitz (all of its eigenvalues have negative real parts)"

        self.Sig0 = solve_continuous_lyapunov(self.F, -self.L @ self.Q @ self.L.T)

        return
    
    def _A(self, dt : float) -> np.ndarray:
        return expm(self.F * dt)
    
    def _Sigma(self, dt : float) -> np.ndarray:
        A = self._A(dt)
        return self.Sig0 - A @ self.Sig0 @ A.T


class MaternProcess(LTISDE):
    """
    The prior GP with Matern Kernel is defined via the stochastic process X(t) as

    X'(t) = F X(t) + L dW(t), 

    where dW(t) is white noise.
    
    n : number of variables (multi-output GP)
    p : number of derivatives (order of the Matern process, or \nu)
    magnitude : magnitude of the process
    lengthscale : lenghtscale of the process

    Note that
    X(t) = (f1, f2, ...,fn, f1', f2', ..., f1^p, ..., fn^p). 
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

        F0 = np.eye(p+1, k=1)
        F0[-1, :] = np.array([- binom(p+1, i) * self.lamb**(p+1 - i) for i in range(p+1)])
        L0 = np.eye(1, p+1, k=p).T

        F = np.kron(F0, np.eye(n))
        L = np.kron(L0, np.eye(n))
        Q = self.spectral_density * np.eye(n) #noise is of the form (w1, ..., wn) for (x1^p, ..., xn^p) only 
        
        super().__init__(n=n, p=p, F=F, L=L, Q=Q)

        return