import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from cvxopt import solvers, matrix

class AbstractSVM:
    def __init__(self):
        pass

    def score(self, X):
        if self.kernel == 'rbf':
            K = - self.gamma * cdist(X, self.data_, 'sqeuclidean')
            np.exp(K, out = K)
        elif self.kernel == 'linear':
            K = np.dot(X, self.data_.T) 
        else:
            raise NotImplementedError
    
        scores = np.zeros((len(X), self.n_class_))
    
        for i in range(self.n_class_-1):
            for j in range(i+1, self.n_class_):
                params = self.results_[i][j-i-1]
                mask1 = self.lbls_ == i
                mask2 = self.lbls_ == j
                mask0 = np.logical_or(mask1, mask2)
                coeffs = params['coeffs'] * (2 * (self.lbls_[mask0] == i) - 1)
                val = params['threshold'] + np.dot(K[:, mask0], coeffs)
                scores[val >= 0, i] += 1
                scores[val <= 0, j] += 1
                
        return scores
    
    def predict(self, X):
        scores = self.score(X)
        return np.argmax(scores, 1)

class NuSVM(AbstractSVM):
    def __init__(self, nu, kernel = 'rbf', gamma = 1.):
        self.nu = nu
        if kernel not in ['rbf', 'linear']:
            raise ValueError("Only RBF and linear kernels are implemented for now. Valid options for the kernel parameter are 'rbf' and 'linear'.")
        else:
            self.kernel = kernel
            if kernel == 'rbf':
                self.gamma = gamma

    def fit(self, X, Y):
        self.data_ = X
        self.lbls_ = Y
    
        if self.kernel == 'linear':
            K = np.dot(X, X.T) # ne pas executer
        elif self.kernel == 'rbf':
            K = - self.gamma * squareform(pdist(X, 'sqeuclidean'))
            np.exp(K, out = K)
        else:
            raise NotImplementedError
        
        n_class = int(np.max(Y) + 1)
        self.n_class_ = n_class
        
        self.results_ = []
        
        for i in range(n_class-1):
            self.results_.append([])
            for j in range(i+1, n_class):
                mask1 = Y == i
                mask2 = Y == j
                mask0 = np.logical_or(mask1, mask2)
                K_small = K[mask0, :]
                K_small = K_small[:, mask0]
                m = np.sum(mask0)
                
                P = K_small.copy()
                P[Y[mask0] == j, :] *= -1
                P[:, Y[mask0] == j] *= -1
                q = np.zeros(m)
                G = np.zeros((2 * m, m))
                G[:m, :] = np.eye(m)
                G[m:, :] = - np.eye(m)
                h = np.zeros(2 * m)
                h[:m] = 1. / m
                A = np.ones((2, m))
                A[0, Y[mask0] == j] = -1
                A[1, :] = 1
                b = np.array([0, self.nu])
                
                result = solvers.qp(P = matrix(P),
                    q = matrix(q),
                    G = matrix(G),
                    h = matrix(h),
                    A = matrix(A),
                    b = matrix(b))
                alpha = np.array(result['x']).flatten()
                grad = np.dot(P, alpha)
                mask1_bis = (alpha > 0) * (alpha < 1./m) * (Y[mask0] == i)
                mask2_bis = (alpha > 0) * (alpha < 1./m) * (Y[mask0] == j)
                val1 = grad[Y[mask0] == i].mean()
                val2 = grad[Y[mask0] == j].mean()
                res_val = np.linalg.solve([[1,-1],[1,1]], [val1, val2])
                
                self.results_[i].append({
                    'coeffs': alpha,
                    'threshold': res_val[1],
                    'margin': res_val[0]})
        
        

class CSVM(AbstractSVM):
    def __init__(self, C = 1., kernel = 'rbf', gamma = 1.):
        self.C = C
        if kernel not in ['rbf', 'linear']:
            raise ValueError("Only RBF and linear kernels are implemented for now. Valid options for the kernel parameter are 'rbf' and 'linear'.")
        else:
            self.kernel = kernel
            if kernel == 'rbf':
                self.gamma = gamma

    def fit(self, X, Y):
        self.data_ = X
        self.lbls_ = Y
    
        if self.kernel == 'linear':
            K = np.dot(X, X.T) # ne pas executer
        elif self.kernel == 'rbf':
            K = - self.gamma * squareform(pdist(X, 'sqeuclidean'))
            np.exp(K, out = K)
        else:
            raise NotImplementedError
        
        n_class = int(np.max(Y) + 1)
        self.n_class_ = n_class
        
        self.results_ = []
        
        for i in range(n_class-1):
            self.results_.append([])
            for j in range(i+1, n_class):
                mask1 = Y == i
                mask2 = Y == j
                mask0 = np.logical_or(mask1, mask2)
                K_small = K[mask0, :]
                K_small = K_small[:, mask0]
                m = np.sum(mask0)
                
                P = K_small.copy()
                P[Y[mask0] == j, :] *= -1
                P[:, Y[mask0] == j] *= -1
                q = - np.ones(m)
                G = np.zeros((2 * m, m))
                G[:m, :] = np.eye(m)
                G[m:, :] = - np.eye(m)
                h = np.zeros(2 * m)
                h[:m] = self.C
                A = np.ones((1, m))
                A[0, Y[mask0] == j] = -1
                b = np.zeros((1, 1))
                
                result = solvers.qp(P = matrix(P),
                    q = matrix(q),
                    G = matrix(G),
                    h = matrix(h),
                    A = matrix(A),
                    b = matrix(b))
                alpha = np.array(result['x']).flatten()
                grad = np.dot(P, alpha)
                mask1_bis = (alpha > 0) * (alpha < 1./m) * (Y[mask0] == i)
                mask2_bis = (alpha > 0) * (alpha < 1./m) * (Y[mask0] == j)
                val1 = grad[Y[mask0] == i].mean()
                val2 = grad[Y[mask0] == j].mean()
                res_val = np.linalg.solve([[1,-1],[1,1]], [val1, val2])
                
                self.results_[i].append({
                    'coeffs': alpha,
                    'threshold': res_val[1],
                    'margin': res_val[0]})
        
        
# vim: set sts=4 ts=4 sw=4 tw=0 et :
