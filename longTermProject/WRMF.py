import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve

class WRMF:
    
    def __init__(self, data, factors = 40, iteration = 30, regulation = 0.8):
        self.data = data
        self.users = data.shape[0]
        self.items = data.shape[1]
        self.factors = factors
        self.iteration = iteration
        self.param = regulation
    
    def iterUser(self, vectors):
        fixed = vectors.shape[0]
        yy = fixed.T.dot(fixed)
        I = sparse.eye(fixed)
        lmbda = self.param*sparse.eye(self.factors)
        answer = np.zeros(self.users*self.factors).reshape(self.user,self.factors)
        for i in range(self.users):
            value = self.data[i].toarray()
            CuI = sparse.diags(value,[0])
            pu = value.copy()
            pu[np.where(pu != 0)] = 1
            yCuIy = fixed.T.dot(CuI).dot(fixed)
            ytCuPu = fixed.T.dot(CuI+I).dot(sparse.csr_matrix(pu).T)
            Xu = spsolve(yy + yCuIy + lmbda, ytCuPu)
            answer[i] = Xu
        return answer

    def iterItems(self,vectors):
        fixed = vectors.shape[0]
        yy = fixed.T.dot(fixed)
        I = sparse.eye(fixed)
        lmbda = self.param*sparse.eye(self.factors)
        answer = np.zeros(self.items*self.factors).reshape(self.items,self.factors)
        for i in range(self.items):
            value = self.data[i].toarray()
            CuI = sparse.diags(value,[0])
            pu = value.copy()
            pu[np.where(pu != 0)] = 1
            yCuIy = fixed.T.dot(CuI).dot(fixed)
            ytCuPu = fixed.T.dot(CuI+I).dot(sparse.csr_matrix(pu).T)
            Xu = spsolve(yy + yCuIy + lmbda, ytCuPu)
            answer[i] = Xu
        return answer

    def train(self):
        
        self.user_vectors = np.random.standard_normal(size = (self.users,self.factors))
        self.item_vectors = np.random.standard_normal(size = (self.items,self.factors))

        for i in range(self.iteration):
            self.user_vectors = self.iterUser(sparse.csr_matrix(self.item_vectors))
            self.item_vectors = self.iterItems(sparse.csr_matrix(self.user_vectors))
        
        self.model = self.user_vectors.dot(self.item_vectors.T)
