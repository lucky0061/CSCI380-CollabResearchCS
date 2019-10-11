
# coding: utf-8

# # Building MAML From Scratch

# In the last section we saw how MAML works. We saw how MAML obtains a better and robust model parameter $\theta$ that is generalizable across tasks.
#
#
# Now we will better understand MAML by coding them from scratch. For better understanding, we consider a simple binary classification task. We randomly generate our input data and we train them with a simple single layer neural network and try to find the optimal parameter theta.
#
# Now we will step by step how exacly we are doing this,
#
# First we import all the necessary libraries,

# In[1]:


import numpy as np


# ## Generate Data Points

# Now we define a function called sample_points for generating our input (x,y) pairs. It takes the parameter k as an input which implies number of (x,y) pairs we want to sample.

# In[2]:


def sample_points(k):
    x = np.random.rand(k,50)
    y = np.random.choice([0, 1], size=k, p=[.5, .5]).reshape([-1,1])
    return x,y


# The above function returns output as follows,

# In[3]:


x, y = sample_points(10)
print(x[0])
print(y[0])


# ## Single Layer Neural Network

# For simplicity and better understand, we use a neural network with only single layer for predicting the output. i.e,
#
# a = np.matmul(X, theta)
#
# YHat = sigmoid(a)
#
#
#
# __*So, we use MAML for finding this optimal parameter value theta that is generalizable across tasks. So that
# for a new task, we can learn from a few data points in a lesser time by taking very less gradient steps.*__

# ## MAML

# Now, we define a class called MAML where we implement the MAML algorithm. In the \__init__  method we will initialize all the necessary variables. Then we define our sigmoid activation function. Followed by we define our train function.

# You can check the comments written above each line of code for understanding.

# In[4]:


class MAML(object):
    def __init__(self):

        #initialize number of tasks i.e number of tasks we need in each batch of tasks
        self.num_tasks = 10

        #number of samples i.e number of shots  -number of data points (k) we need to have in each task
        self.num_samples = 10

        #number of epochs i.e training iterations
        self.epochs = 10000

        #hyperparameter for the inner loop (inner gradient update)
        self.alpha = 0.0001

        #hyperparameter for the outer loop (outer gradient update) i.e meta optimization
        self.beta = 0.0001

        #randomly initialize our model parameter theta
        self.theta = np.random.normal(size=50).reshape(50, 1)

    #define our sigmoid activation function
    def sigmoid(self,a):
        return 1.0 / (1 + np.exp(-a))


    #now let us get to the interesting part i.e training :P
    def train(self):

        #for the number of epochs,
        for e in range(self.epochs):

            self.theta_ = []

            #for task i in batch of tasks
            for i in range(self.num_tasks):

                #sample k data points and prepare our train set
                XTrain, YTrain = sample_points(self.num_samples)

                a = np.matmul(XTrain, self.theta)

                YHat = self.sigmoid(a)

                #since we are performing classification, we use cross entropy loss as our loss function
                loss = ((np.matmul(-YTrain.T, np.log(YHat)) - np.matmul((1 -YTrain.T), np.log(1 - YHat)))/self.num_samples)[0][0]

                #minimize the loss by calculating gradients
                gradient = np.matmul(XTrain.T, (YHat - YTrain)) / self.num_samples

                #update the gradients and find the optimal parameter theta' for each of tasks
                self.theta_.append(self.theta - self.alpha*gradient)


            #initialize meta gradients
            meta_gradient = np.zeros(self.theta.shape)

            # Line 8
            for i in range(self.num_tasks):

                #sample k data points and prepare our test set for meta training
                XTest, YTest = sample_points(10)

                #predict the value of y
                a = np.matmul(XTest, self.theta_[i])

                YPred = self.sigmoid(a)

                #compute meta gradients
                meta_gradient += np.matmul(XTest.T, (YPred - YTest)) / self.num_samples


            #update our randomly initialized model parameter theta with the meta gradients
            self.theta = self.theta-self.beta*meta_gradient/self.num_tasks

            if e%1000==0:
                print("Epoch {}: Loss {}\n".format(e,loss))
                print('Updated Model Parameter Theta\n')
                print('Sampling Next Batch of Tasks \n')
                print('---------------------------------\n')


# In[5]:


model = MAML()


# In[6]:


model.train()
