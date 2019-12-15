


# imports

import numpy as np
import random
import pickle
import os 


# activation function

def perceptron(z):
    return -1 if z<=0 else 1

# loss functions

def ploss(yhat, y):
    return max(0, -yhat*y)


# In[51]:


class Sonar_Model:
    
    
    def __init__(self, dimension=None, weights=None, bias=None, activation=(lambda x: x)):
    
        self._dim = dimension
        self.w = weights or np.random.normal(size=self._dim)
        self.w = np.array(self.w)
        self.b = bias if bias is not None else np.random.normal()
        self._a = activation
        #self.predict = predict.__get__(self)
    
    def __str__(self):
        
        return "Simple cell neuron\n        \tInput dimension: %d\n        \tBias: %f\n        \tWeights: %s\n        \tActivation: %s" % (self._dim, self.b, self.w, self._a.__name__)
    
     #Sonar class should have a predict(v) method that uses internal weights to make prediction on new data 
    
    def __call__(self, v):

        yhat = self._a(np.dot(self.w, np.array(v)) + self.b)
        return yhat
    
    def ppredict(self, x):
        return self(x)

    
    def load_model(self, file_path):
        
        #open the pickle file and update the model's parameters
        
#open the pickle file and update the model's parameters
        #// Deserialize a model
        with open(file_path, mode='rb') as f:
            file=pickle.load(f)
            
        self._dim = file._dim
        self.w = file.w
        self.b = file.b
        self._a = file._a



    def save_model(self):
        
        #save your model as 'cat_model.pkl' in the local path
        #// Serialize a model
        #relative_path = '/Users/guillaumedelande/Documents/AIGroupWork/91931144/sonar_model.pkl'
        f= open('sonar_model.pkl','wb')
        pickle.dump(self, f)
        f.close
   

# In[52]:


class Sonar_Trainer:
    
    def __init__(self, dataset, model):
        
        self.dataset = dataset
        self.model = model
        self.loss = ploss

    def accuracy(self, data):
        '''
        return the accuracy on data given data iterator
        '''
        acc = 100*np.mean([1 if self.model.ppredict(x) == y else 0 for x, y in data])
        return acc
    
    
    
  #Sonar class should have a public method train, which trains the perceptron on loaded data
    def train(self, lr, ne):
        '''
        This method should:
        1. display initial accuracy on the training data loaded in the constructor
        2. update parameters of the model instance in a loop for ne epochs using lr learning rate
        3. display final accuracy
        '''
        
        print("training model on data...")
        accuracy = self.accuracy(self.dataset)
        print("initial accuracy: %.3f" % (accuracy))
        
        for epoch in range(ne+1):
            self.dataset._shuffle()
            for d in self.dataset:
                x, y = d
                x = np.array(x)
                yhat = self.model(x)
                error = y - yhat
                self.model.w += lr*(y-yhat)*x  
                self.model.b += lr*(y-yhat)
            accuracy = self.accuracy(self.dataset)    
            print('>epoch=%d, learning_rate=%.3f, accuracy=%.3f' % (epoch+1, lr, accuracy))
            
        print("training complete")
        
        #Train method returns a single float representing the mean squarre error on the trained set
        print("final accuracy: %.3f" % (accuracy))


# In[55]:


class Sonar_Data:


#Sonar Class should have the datafile relative path (string) and name (string) as contrustor arguments
        
    def __init__(self, relative_path, data_file_name):
        '''
        initialize self.index; load and preprocess data; shuffle the iterator
        '''
        self.index = -1
        full_path = os.path.join(relative_path,data_file_name)
        self.raw = pickle.load(open(full_path,'rb'))
        self.simple = [(list(d), -1) for d in self.raw['r']]+[(list(d), 1) for d in self.raw['m']]
        
    def __iter__(self):
        '''
        See example code (ngram) in lecture slides
        '''
        return self

    def __next__(self):
        '''
        See example code (ngram) in slides
        '''
        self.index += 1
  
        if self.index == len(self.simple):
            self.index = -1
            raise StopIteration
        
        return self.simple[self.index][0], self.simple[self.index][1]

    def _shuffle(self):
        '''
        shuffle the data iterator
        '''
        return random.shuffle(self.simple)
       
    


# In[94]:


def main():

    data = Sonar_Data(relative_path='/Users/guillaumedelande/Documents/AIGroupWork/stephenfitz.keio2019aia/keio2019aia/data/assignment1/', data_file_name='sonar_data.pkl')
    model = Sonar_Model(dimension=60, activation=perceptron)  # specify the necessary arguments
    trainer = Sonar_Trainer(data, model)
    trainer.train(0.1,300) # experiment with learning rate and number of epochs
    model.save_model()


if __name__ == '__main__':

    main()


# In[ ]:




