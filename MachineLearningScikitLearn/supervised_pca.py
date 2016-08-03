#supervised PCA according to Supervised Principal Compontent Anaysis by Ghodsi et al. 2010

import numpy as np
from scipy import linalg

from ..utils.arpack import eigsh
from ..base import BaseEstimator, TransformerMixin
from ..preprocessing import KernelCenterer, scale
from ..metrics.pairwise import pairwise_kernels


from time import clock



class SupervisedPCA(BaseEstimator, TransformerMixin):
    """Supervised Principal component analysis (SPCA)
    Non-linear dimensionality reduction through the use of kernels.
    Parameters
    ----------
    n_components: int or None
        Number of components. If None, all non-zero components are kept.
    kernel: "linear" | "poly" | "rbf" | "sigmoid" | "precomputed"
        Kernel.
        Default: "linear"
    degree : int, optional
        Degree for poly, rbf and sigmoid kernels.
        Default: 3.
    gamma : float, optional
        Kernel coefficient for rbf and poly kernels.
        Default: 1/n_features.
    coef0 : float, optional
        Independent term in poly and sigmoid kernels.
    eigen_solver: string ['auto'|'dense'|'arpack']
        Select eigensolver to use.  If n_components is much less than
        the number of training samples, arpack may be more efficient
        than the dense eigensolver.
    tol: float
        convergence tolerance for arpack.
        Default: 0 (optimal value will be chosen by arpack)
    max_iter : int
        maximum number of iterations for arpack
        Default: None (optimal value will be chosen by arpack)
    Attributes
    ----------
    `lambdas_`, `alphas_`:
        Eigenvalues and eigenvectors of the centered kernel matrix
    """

    def __init__(self, n_components=None, kernel="linear", gamma=None, degree=3,
                 coef0=1, fit_inverse_transform=False,
                 eigen_solver='auto', tol=0, max_iter=None):


        self.n_components = n_components
        self.kernel = kernel.lower()
        self.gamma = None
        self.degree = degree
        self.coef0 = coef0
        self.fit_inverse_transform = fit_inverse_transform
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.centerer = KernelCenterer()

    def transform(self, X):
        """
        Returns a new X, X_trans, based on previous self.fit() estimates
        """
        return(np.dot(X,self.alphas_))


    def fit(self,X,Y):
        """Returns (gen. eigenvalues, gen. eigenvectors) of matrix Q=KxKyKx.
        The snapshots in X and Y are stores by rows.

        Parameters
        ----------
            X: (n_samples,n_inputs) np.ndarray
                Matrix of normalized input snapshots
            Y: (n_samples,n_outputs) np.ndarray
                Matrix of normalized output snapshots

        Returns:
        --------
            lambdas: (n_components,) np.ndarray
                Array of eigenvalues of XKyX^t matrix stored by decreasing
                order. Only eigenvalues >= 0 are conserved.
                The number of components is at most equal to the rank of X.
            alphas: (n_inputs,n_components) np.ndarray
                Array containing the eigenvectors associated with the eigenva
                -lues in lambdas stored by columns. It is the matrix of new
                basis vectors coordinates in the original basis.
        Info
        ----
        Solves the generalized eigenvalues problem to build the spca coeffs as
        described in algo. 1 from Ghodsi 2010 paper.
        """
        if Y.ndim == 1:
            Y.reshape((-1,1))
        assert X.shape[0] == Y.shape[0], "X and Y don't contain the same\
 numbers of samples. Please check the matrices."
        self._fit(X,Y)
        return(self.lambdas_,self.alphas_)

    def fit_transform( self, X, Y):
        """Fits the SPCA model (solves eigenvalues problem) and returns trans
        -formed input variables.

        Parameters
        ----------
            X: (n_samples,n_inputs) np.ndarray
                Matrix of normalized input snapshots
            Y: (n_samples,n_outputs) np.ndarray
                Matrix of normalized output snapshots

        Returns:
        --------
            X_fit_trans: (nsamples,n_components) np.ndarray
                Matrix of coordinates of input samples used for fit projected
                on the space spanned by the eigenvectors computed during the
                fit. The coordinates are expressed in the basis defined by the
                eigenvectors.
        """
        self.fit( X,Y)
        return(self._transform())

    def _transform(self):
        """Returns the transformed version of input variables used for fit.
        """
        return(np.dot(self.X_fit,self.alphas_))


    def _fit(self, X, Y):
        #store the input X and Y used for fit
        self.X_fit = X
        self._nfeature = X.shape[1]
        #Compute kernel matrix of Y
        Ky = self.centerer.fit_transform(self._get_kernel(Y))
        #maximum number of components that can be kept is the min between the
        #number of features and samples.
        n_comp_max = min(X.shape)
        if self.n_components is None:
            n_components = n_comp_max
        else:
            n_components = min(n_comp_max, self.n_components)
        #compute eigenvalues and eigenvetors of XTKX^T
        #---------------------------------------------
        M = (X.T).dot(Ky).dot(X)
        # Choosing eigensolver
        if self.eigen_solver == 'auto':
                if M.shape[0] > 200 and n_components < 10:
                    eigen_solver = 'arpack'
                else:
                    eigen_solver = 'dense'
        else:
            eigen_solver = self.eigen_solver
        #Eigen-problem resolution
        if eigen_solver == 'dense':
            index_eigv_min = M.shape[0] - n_components
            index_eigv_max = M.shape[0] - 1
            self.lambdas_, self.alphas_ = linalg.eigh(M,
                                                      eigvals=(index_eigv_min,
                                                               index_eigv_max))
        elif eigen_solver == 'arpack':
                self.lambdas_, self.alphas_ = eigsh(M, n_components,
                                                    which="LA",
                                                    tol=self.tol)
        #-----------------------------------------------------------
        #Sorting eigenvalues and eigenvectors by decreasing order of
        #eigenvalues
        indices = self.lambdas_.argsort()[::-1]
        self.lambdas_ = self.lambdas_[indices]
        self.alphas_ = self.alphas_[:, indices]
        #remove the zero/negative eigenvalues
        self.alphas_ = self.alphas_[:, self.lambdas_ > 0 ]
        self.lambdas_ = self.lambdas_[ self.lambdas_ > 0 ]
        return()


    def _get_kernel(self, X, Y=None):
        params = {"gamma": self.gamma or 1/self._nfeature,
                  "degree": self.degree,
                  "coef0": self.coef0}
        try:
            return pairwise_kernels(X, Y, metric=self.kernel,
                                    filter_params=True,  n_jobs = -1, **params)
        except AttributeError:
            raise ValueError("%s is not a valid kernel. Valid kernels are: "
                             "rbf, poly, sigmoid, linear and precomputed."
                             % self.kernel)

        
         
class KernelSupervisedPCA(BaseEstimator, TransformerMixin):
    """Kernel Supervised Principal component analysis (SPCA)
    Non-linear dimensionality reduction through the use of kernels.

    Parameters
    ----------
    n_components: int or None
        Number of components. If None, all non-zero components are kept.
    x||ykernel: "linear" | "poly" | "rbf" | "sigmoid" | "precomputed"
        Kernel.
        Default: "linear"
    degree : int, optional
        Degree for poly, rbf and sigmoid kernels.
        Default: 3.
    gamma : float, optional
        Kernel coefficient for rbf and poly kernels.
        Default: 1/n_features.
    coef0 : float, optional
        Independent term in poly and sigmoid kernels.
    eigen_solver: string ['auto'|'dense'|'arpack']
        Select eigensolver to use.  If n_components is much less than
        the number of training samples, arpack may be more efficient
        than the dense eigensolver.
    tol: float
        convergence tolerance for arpack.
        Default: 0 (optimal value will be chosen by arpack)
    max_iter : int
        maximum number of iterations for arpack
        Default: None (optimal value will be chosen by arpack)
    Attributes
    ----------
    `lambdas_`, `alphas_`:
        Eigenvalues and eigenvectors of the centered kernel matrix
    """

    def __init__(self, n_components=None, xkernel={'kernel': "linear", 'gamma':0, 'degree':3,
                 'coef0':1}, ykernel = {'kernel': "linear", 'gamma':0, 'degree':3,
                 'coef0':1},  fit_inverse_transform=False,
                 eigen_solver='auto', tol=0, max_iter=None):
        self.n_components = n_components
        self.xkernel = xkernel
        self.ykernel = ykernel
        self.fit_inverse_transform = fit_inverse_transform
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.centerer = KernelCenterer()


    def transform(self, X):
        """
        Returns a new X, X_trans, based on previous self.fit() estimates.
        """
        K = self.centerer.fit_transform(self._get_kernel(self.X_fit, self.xkernel, X))
        X_trans = np.dot(self.alphas_.T,K)
        return(X_trans)


    def fit(self,X,Y):
        """Returns (gen. eigenvalues, gen. eigenvectors) of matrix Q=KxKyKx.
        The snapshots in X and Y are stores by rows.

        Parameters
        ----------
            X: (n_samples,n_inputs) np.ndarray
                Matrix of normalized input snapshots
            Y: (n_samples,n_outputs) np.ndarray
                Matrix of normalized output snapshots
        Info
        ----
        Solves the generalized eigenvalues problem to build the kspca coeffs as
        described in algo. 3 from Ghodsi 2010 paper.
        """
        self._fit(X,Y)
        return(self.lambdas_,self.alphas_)

    def fit_transform( self, X, Y):
        """Fits the KSPCA model (solves eigenvalues problem) and returns trans
        -formed input variables.
        """
        self.fit( X,Y)
        return(self._transform())

    def _transform(self):
        """Returns the transformed version of input variables used for fit.
        """
        return(self.Kx_fit.dot(self.alphas_))


    def _fit(self, X, Y):
        #find kenerl matrices of X and Y
        Ky = self.centerer.fit_transform(self._get_kernel(Y, self.ykernel))
        Kx = self.centerer.fit_transform( self._get_kernel(X, self.xkernel))
        #stores the X, Kx and Ky matrices used for fit
        self.X_fit = X
        self.Kx_fit = Kx
        self.Ky_fit = Ky
        #n_components is set as the min between the specified number of compo-
        #nents and the number of samples
        if self.n_components is None:
            n_components = Ky.shape[0]
        else:
            n_components = min(Ky.shape[0], self.n_components)
        #--------------------------------------------------------------
        #Compute generalized eigenvalues and eigenvectors of Kx^T.Ky.Kx
        M = (Kx).dot(Ky).dot(Kx)

        #Chose the eigensolver to be used
        if self.eigen_solver == 'auto':
            if M.shape[0] > 200 and n_components < 10:
                eigen_solver = 'arpack'
            else:
                eigen_solver = 'dense'
        else:
            eigen_solver = self.eigen_solver
        #Solve the generalized eigenvalues problem
        if eigen_solver == 'dense':
            self.lambdas_, self.alphas_ = linalg.eigh(
                M, Kx, eigvals=(M.shape[0] - n_components, M.shape[0] - 1))
        elif eigen_solver == 'arpack':
            self.lambdas_, self.alphas_ = eigsh(A=M, M=Kx, k=n_components,
                                                which="LA",
                                                tol=self.tol)
        else:
            #useless for now
            self.lambdas_, self.alphas_ = self.eigen_solver(M, Kx, n_components)

        #Sort the eigenvalues in increasing order
        indices = self.lambdas_.argsort()[::-1]
        self.lambdas_ = self.lambdas_[indices]
        self.alphas_ = self.alphas_[:, indices]

        #remove the zero/negative eigenvalues
        self.alphas_ = self.alphas_[:, self.lambdas_ > 0 ]
        self.lambdas_ = self.lambdas_[ self.lambdas_ > 0 ]
        return()

    def _get_kernel(self, X, params, Y=None):
        """Computes and returns kernel of X with given params.
        If Y is specified, then returns the pairwise kernel between X and Y.
        """
        try:
            coparams = copy.copy(params)
            return pairwise_kernels(X, Y, metric=coparams.pop('kernel'),
                                     n_jobs = 1, **coparams)
        except AttributeError:
            raise ValueError("%s is not a valid kernel. Valid kernels are: "
                             "rbf, poly, sigmoid, linear and precomputed."
                             % params['kernel'])
        
    
