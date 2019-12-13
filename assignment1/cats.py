#!/usr/bin/env python
# coding: utf-8
#Guillaume Delande - Keio Univeristy 2019-2020 - AI Assignment 1


# # INTRODUCTION             

# In order to help you with the first assignment, this file provides a general outline of your program. You will implement the details of various pieces of Python code grouped in functions. Those functions are called within the main function, at the end of this source file. Please refer to the lecture slides for the background behind this assignment. You will submit three python files (sonar.py, cat.py, digits.py) and three pickle files (sonar_model.pkl, cat_model.pkl, digits_model.pkl) which contain trained models for each tasks.
# Good luck!

# # CODE


########################################
#### I M P O R T ######################
import numpy as np
import pickle
import random
import matplotlib.pyplot as plt #not sure I used it
#%matplotlib inline
import math #not sure I used it
import os
import pprint
########################################
########################################


########################################
#### FUNCTIONS #########################

##### ACTIVATION FUNCTION
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


##### LOSS FUNCTION
def lrloss(yhat, y):
    return 0.0 if yhat==y else -1.0*(y*np.log(yhat)+(1-y)*np.log(1-yhat))
#Corresponds to the Local vs Global Optima


########################################
#### CLASSES #########################
file_path='/Users/guillaumedelande/Documents/AIGroupWork/stephenfitz.keio2019aia/keio2019aia/data/assignment1/cat_data.pkl'
class Cat_Model:

    def __init__(self, dimension, weights=None, bias=None, activation=(lambda x: x)):
        self._dim = dimension
        self.w = weights or np.random.normal(size=self._dim)
        self.w = np.array(self.w)
        self.b = bias if bias is not None else np.random.normal()
        self._a = activation
        #self.predict = predict.__get__(self)

        
    ##### PREDICTION FUNCTION
    def lrpredict(self, x):
        return 1 if self(x)>0.5 else 0
    
    
    def __str__(self):
        
        info = "Simple cell neuron\n        \tInput dimension: %d\n        \tBias: %f\n        \tWeights: %s\n        \tActivation: %s" % (self._dim, self.b, self.w, self._a.__name__)
        return info

    
    def __call__(self, x):
        
        #return the output of the network
        
        #yhat = self._a(np.dot(self.w, np.array(x)) + self.b) --> perceptron
        yhat = self._a(np.dot(self.w, np.reshape(x, x.size)) + self.b)
        return yhat

####NEED TO DO 
#https://www.geeksforgeeks.org/saving-a-machine-learning-model/
#Pickle string: The pickle module implements a fundamental, but powerful algorithm for serializing and de-serializing a Python object structure. 
#pickle.dump to serialize an object hierarchy, you simply use dump().
#pickle.load to deserialize a data stream, you call the loads() function.
    def load_model(self, file_path):
        
        #open the pickle file and update the model's parameters
        #// Deserialize a model
        with open('/Users/guillaumedelande/Documents/AIGroupWork/91931144/cat_model.pkl', mode='rb') as f:
            a=pickle.load(f)
        
        self._dim = a._dim
        self.w = a.w
        self.b = a.b
        self._a = a._a
        

    def save_model(self):
        
        #save your model as 'cat_model.pkl' in the local path
        #// Serialize a model
        #relative_path = '/Users/guillaumedelande/Documents/AIGroupWork/91931144/cat_model.pkl'
        f= open('cat_model.pkl','wb')
        pickle.dump(self, f)
        f.close
        


class Cat_Trainer:

    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model
        self.loss = lrloss

        
    def accuracy(self, data):
        
        #return the accuracy on data given data iterator
        
        acc = 100*np.mean([1 if self.model.lrpredict(x) == y else 0 for x, y in data])
        #print(acc)
        return acc

    
    def train(self, lr, ne):
        
        #This method should:
        #1. display initial accuracy on the training data loaded in the constructor
        
        print("training model on data...")
        accuracy = self.accuracy(self.dataset) #.samples?
        print("initial accuracy: %.3f" % (accuracy))
                
        
        #2. update parameters of the model instance in a loop for ne epochs using lr learning rate
        costs=[]
        accuracies=[]
        
        for epoch in range(1, ne+1):
            
##############################
            J=0
            dw=0 #NOT SURE it's called dx
            for d in self.dataset: #.samples               
                #print(d[0])
                Cat_Data.shuffle(self, self.dataset.samples)
                #print(d)
                xi, yi = d
                #x = np.array(x)
                yhat = self.model(xi)
                print("y: "+str(yi)+", \t yhat: "+str(yhat)+", \t self.dataset.index: "+str(self.dataset.index))
                J += self.loss(yhat, yi)
                dy = yhat - yi
                dw += xi*dy
                #error = y - yhat
                #self.model.w += lr*(y-yhat)*x
                #self.model.b += lr*(y-yhat)
            J /= len(self.dataset.samples)
            dw /= len(self.dataset.samples)
            self.model.w= self.model.w - lr*dw
            
            accuracy = self.accuracy(self.dataset)
            print('>epoch=%d, learning_rate=%.3f, accuracy=%.3f' % (epoch+1, lr, accuracy))  
            costs.append(J)
            accuracies.append(accuracy)
            
##############################

        #3. display final accuracy
        #and eventually costs
        print("training complete")
        print("final accuracy: %.3f" % (self.accuracy(self.dataset)))



relative_path = '/Users/guillaumedelande/Documents/AIGroupWork/stephenfitz.keio2019aia/keio2019aia/data/assignment1/'
data_file_name = 'cat_data.pkl' 


class Cat_Data():
    def __init__(self, relative_path='/Users/guillaumedelande/Documents/AIGroupWork/stephenfitz.keio2019aia/keio2019aia/data/assignment1/', data_file_name='cat_data.pkl'):
        
########initialize self.index; 
        self.index = -1 #index of image, or image number. 
        
######## Load and preprocess data;
        
    ###Load data
        self.relative_path=relative_path
        self.data_file_name=data_file_name
        
        full_path = os.path.join(relative_path,data_file_name)
        cat_data = pickle.load(open(full_path,'rb'))
        #The pickle file is now loaded
        
        #print(cat_data['train']['cat'].keys())
        self.samples = [(np.reshape(vector, vector.size), 1) for vector in self.standardize(cat_data['train']['cat'])]+[(np.reshape(vector, vector.size),0) for vector in self.standardize(cat_data['train']['no_cat'])]
        
        #Without standardization:
        #self.samples = [(np.reshape(vector, vector.size), 1) for vector in cat_data['train']['cat']]+[(np.reshape(vector, vector.size),0) for vector in cat_data['train']['no_cat']]

        self.shuffle(self.samples)
        
        print("Length of self.samples: "+str(len(self.samples)))
        print("self.samples from 0  to 5: \n"+str(self.samples[0:5]))
        print("Value of y for self.index = 0: "+str(self.samples[self.index][1]))
        print("First value (standardized) of x in the vector where self.index=0: "+str(self.samples[208][0][0]))
        print("Note: The value is different each time because of the shuffle.")
        print('--------------------------------------------------------------')

    def standardize (self, rgb_images):
        mean = np.mean(rgb_images, axis=(1,2), keepdims=True)
        std = np.std(rgb_images, axis=(1,2), keepdims=True)
        
        #standardized_cat=standardize(cat_data['train']['cat'])[0]
        return (rgb_images - mean) / std

    def shuffle(self, a):
        return random.shuffle(a)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        self.index +=1
        if self.index > (len(self.samples)-1): #Length is 209 but we start at zero so we can go up to 208 only
            self.index =-1
            raise StopIteration
            
        return self.samples[self.index][0], self.samples[self.index][1]
        


def main():

    data = Cat_Data(relative_path='/Users/guillaumedelande/Documents/AIGroupWork/stephenfitz.keio2019aia/keio2019aia/data/assignment1/', data_file_name='cat_data.pkl')
    
    #data=preprocess_data(relative_path='/Users/guillaumedelande/Documents/AIGroupWork/stephenfitz.keio2019aia/keio2019aia/data/assignment1/', data_file_name='cat_data.pkl')
    model = Cat_Model(dimension=(64*64*3), weights=None, bias=None, activation=sigmoid)  # specify the necessary arguments    

    trainer = Cat_Trainer(data, model)
    trainer.train(lr=0.01, ne=500) # experiment with learning rate and number of epochs
    #As defined earlier: train(lr, ne) where lr is the learning rate and ne the number of epochs (for the gradient), tweak those values
    model.save_model()
    #print(data[0][0])

if __name__ == '__main__':

    main()


# # Finished
