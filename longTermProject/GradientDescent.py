import numpy as np

class GradientDescent:

    def __init__(self,data,crit = 0.005,learning_rate = 0.01,regular = 0.01,factor = 50):
    
        self.data = data
        self.users = data.shape[0]
        self.items = data.shape[1]
        self.factor = factor
        self.iter = iter
        self.crit = crit
        self.learning_rate = learning_rate
        self.regular = regular
        self.user_vectors = np.random.normal(size=(self.users,self.factor))
        self.item_vectors = np.random.normal(size=(self.items,self.factor))
        self.prediction = self.user_vectors.dot(np.transpose(self.item_vectors))
        self.user_bias = np.zeros(self.users)
        self.item_bias = np.zeros(self.items)
        self.global_bias = 1
        self.mask = np.where(self.data != 0)

    def globalBiasSet(self):
        self.global_bias = np.mean(self.data[self.mask])

    def predict(self,x,y):
        prediction = self.global_bias + self.user_bias[x] + self.item_bias[y] + self.user_vectors[x,:].dot(np.transpose(self.item_vectors[y,:]))
        return prediction

    def iteration(self):

        for i in range(self.users):
            for j in range(self.items):
                if self.data[i,j] > 0:
                    prediction = self.predict(i,j)
                    error = self.data[i,j] - prediction
                    
                    self.user_bias[i] += self.learning_rate * (error - self.regular * self.user_bias[i])
                    self.item_bias[j] += self.learning_rate * (error - self.regular * self.item_bias[j])
                    
                    du = (error * self.item_vectors[j,:]) - (self.regular * self.user_vectors[i,:])
                    di = (error * self.user_vectors[i,:]) - (self.regular * self.item_vectors[j,:])

                    self.user_vectors[i,:] += self.learning_rate * du
                    self.item_vectors[j,:] += self.learning_rate * di
        
        self.prediction = self.global_bias + self.user_vectors.dot(np.transpose(self.item_vectors)) + self.user_bias[:,np.newaxis] + self.item_bias[np.newaxis,:]

        cost = 0
        count = 0
        for i in range(self.users):
            for j in range(self.items):
                if self.data[i,j] > 0:
                    count += 1
                    cost += np.power(self.data[i,j] - self.prediction[i,j],2)
        cost = np.sqrt(cost)
        cost /= len(self.mask[0])

        cost = np.sqrt(cost/count)

        self.cost = cost


    def train(self):
        
        self.globalBiasSet()
        
        while True:

            self.iteration()
            
            if self.cost < self.crit:
                self.model = self.prediction
                break