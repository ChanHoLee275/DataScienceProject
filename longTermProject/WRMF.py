import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve

# 참고 : https://math.stackexchange.com/questions/1072451/analytic-solution-for-matrix-factorization-using-alternating-least-squares/1073170#1073170

class WRMF:
    
    def __init__(self, data, factors = 30, iteration = 40, regulation = 0.02):
        self.data = sparse.csr_matrix(data)
        self.users = data.shape[0]
        self.items = data.shape[1]
        self.factors = factors
        self.iteration = iteration
        self.param = regulation
    
    def iterUser(self, vectors):
        fixed = vectors.shape[0]
        I = sparse.eye(fixed)
        lmbda = self.param*sparse.eye(self.factors)
        answer = np.zeros(self.users*self.factors).reshape(self.users,self.factors)
        for i in range(self.users):
            value = self.data[i].toarray()
            pu = value.copy()
            pu[np.where(pu != 0)] = 1.0
            CuI = sparse.diags(value,[0])
            yCuIy = vectors.T.dot(CuI).dot(vectors)
            ytCuPu = vectors.T.dot(CuI+I).dot(sparse.csr_matrix(pu).T)
            Xu = spsolve(yCuIy + lmbda, ytCuPu)
            answer[i] = Xu
        return answer

    def iterItems(self,vectors):
        fixed = vectors.shape[0]
        I = sparse.eye(fixed)
        lmbda = self.param*sparse.eye(self.factors)
        answer = np.zeros(self.items*self.factors).reshape(self.items,self.factors)
        for i in range(self.items):
            value = self.data[:,i].T.toarray()
            pu = value.copy()
            pu[np.where(pu != 0)] = 1
            CuI = sparse.diags(value,[0])
            yCuIy = vectors.T.dot(CuI).dot(vectors)
            ytCuPu = vectors.T.dot(CuI+I).dot(sparse.csr_matrix(pu).T)
            Xu = spsolve(yCuIy + lmbda, ytCuPu)
            answer[i] = Xu
        return answer

    def train(self):
        
        self.user_vectors = np.random.standard_normal(size = (self.users,self.factors))
        self.item_vectors = np.random.standard_normal(size = (self.items,self.factors))


        for i in range(self.iteration):
            self.user_vectors = self.iterUser(sparse.csr_matrix(self.item_vectors))
            self.item_vectors = self.iterItems(sparse.csr_matrix(self.user_vectors))
        
        self.model = self.user_vectors.dot(self.item_vectors.T)
