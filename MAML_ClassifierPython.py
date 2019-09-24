import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('A.csv')
df = df.drop(df.columns[0], axis =1)

# def sample_points(k, l):
#     x1 = df.loc[k:l-1,['A.Open', 'A.High', 'A.Low', 'A.Close', 'A.Volume', 'A.Adjusted']]
#     y1 = df.loc[k:l-1,['Y']]
#     x2 = np.array(x1.values.tolist())
#     y2 = np.array(y1.values.tolist())
#     scaler0 = MinMaxScaler()
#     scaler1 = MinMaxScaler()
#     scaler0.fit(x2)
#     scaler1.fit(y2)
#     x = scaler0.transform(x2)
#     y = scaler1.transform(y2)
#     return x,y

# Temporary
def sample_points_test():
    k = random.randint(0,14)
    l = k + 5
    x1 = df.loc[k:l-1,['A.Close']]
    y1 = df.loc[k:l-1,['Y']]
    x2 = np.array(x1.values.tolist())
    y2 = np.array(y1.values.tolist())
    scaler0 = MinMaxScaler()
#     scaler1 = MinMaxScaler()
    scaler0.fit(x2)
#     scaler1.fit(y2)
#     x = np.array([x_ if random.random > .1 else random.random for x_ in scaler0.transform(x2)])
#     y = np.array([y_ if random.random > .1 else random.choice([0, 1]) for y_ in scaler1.transform(y2)])
    return x,y

# # Temporary
# def sample_points_train():
#     k = 0
#     l = 20
#     x1 = df.loc[k:l-1,['A.Close']]
#     y1 = df.loc[k:l-1,['Y']]
#     x2 = np.array(x1.values.tolist())
#     y2 = np.array(y1.values.tolist())
#     scaler0 = MinMaxScaler()
# #     scaler1 = MinMaxScaler()
#     scaler0.fit(x2)
# #     scaler1.fit(y2)
#     x = np.array([x_ if(random.random() > .1) else random.random for x_ in scaler0.transform(x2)])
# #     y = np.array([y_ if random.random > .1 else random.choice([0, 1]) for y_ in scaler1.transform(y2)])
#     y = y2
#     return x,y

def mkWindows(train_size, meta_size, test_size, data_length, shift = 0):
    index = 0
    windows = []
    window_type = 0 #0=train, 1=meta, 2=test
    while index+train_size+meta_size+test_size < data_length:
        if window_type==0: 
            windows += [(index, index+train_size)]
            window_type = 1
        elif window_type==1:
            windows += [(index+train_size, index+train_size+meta_size)]
            window_type = 2
        else: 
            windows += [(index+train_size+meta_size, index+train_size+meta_size+test_size)]
            index += shift
            window_type = 0
            
    return windows

class MAML(object):
    def __init__(self):
        
        #initialize number of tasks i.e number of tasks we need in each batch of tasks
        self.num_tasks = 5
        
        #number of samples i.e number of shots  -number of data points (k) we need to have in each task
        self.num_train_samples = 20
        self.num_meta_samples = 5
        self.num_test_samples = 5
        self.data_length = 20
        self.shift = 5

        #number of epochs i.e training iterations
        self.epochs = 100
        
        #hyperparameter for the inner loop (inner gradient update)
        self.alpha = 0.0001
        
        #hyperparameter for the outer loop (outer gradient update) i.e meta optimization
        self.beta = 0.0001
       
        #randomly initialize our model parameter theta
        self.theta = np.random.normal(size=1).reshape(1, 1)
        
        self.windows = mkWindows(self.num_train_samples,self.num_meta_samples,
                                 self.num_test_samples,self.data_length,self.shift)
      
    #define our sigmoid activation function  
    def sigmoid(self,a):
        return 1.0 / (1 + np.exp(-a))
    
    def classify(self,value):
        return 1 if value > 0.5 else 0    
    
    #now let us get to the interesting part i.e training :P
    def train(self):
        for wi in range(0,len(self.windows),3):
            #for the number of epochs,
            for e in range(self.epochs):        

                self.theta_ = []

                #for task i in batch of tasks
                for i in range(self.num_tasks):

                    #sample k data points and prepare our train set
#                     XTrain, YTrain = sample_points(*self.windows[wi])
                    XTrain, YTrain = sample_points_test()
    #                 XTrain = X_train[:20]
    #                 YTrain = Y_train[:20]

                    a = np.matmul(XTrain, self.theta)
    #                 print(a)

                    YHat = self.sigmoid(a)
    #                 print(YHat)

                    #since we are performing classification, we use cross entropy loss as our loss function
                    loss = ((np.matmul(-YTrain.T, np.log(YHat)) - np.matmul((1 -YTrain.T), np.log(1 - YHat)))/self.num_train_samples)[0][0]
    #                 print(loss)
                    #minimize the loss by calculating gradients
                    gradient = np.matmul(XTrain.T, (YHat - YTrain)) / self.num_train_samples

                    #update the gradients and find the optimal parameter theta' for each of tasks
                    self.theta_.append(self.theta - self.alpha*gradient)


                #initialize meta gradients
                meta_gradient = np.zeros(self.theta.shape)

                for i in range(self.num_tasks):

                    #sample k data points and prepare our test set for meta training
#                     XMeta, YMeta = sample_points(*self.windows[wi+1])
                    XMeta, YMeta = sample_points_test()
    #                 XTest = X_train[20:23]
    #                 YTest = Y_train[20:23]


                    #predict the value of y
                    a = np.matmul(XMeta, self.theta_[i])

                    YPred = self.sigmoid(a)

                    #compute meta gradients
                    meta_gradient += np.matmul(XMeta.T, (YPred - YMeta)) / self.num_meta_samples


                #update our randomly initialized model parameter theta with the meta gradients
                self.theta = self.theta-self.beta*meta_gradient/self.num_tasks
#             print("THeta: {}\n".format(self.theta))  
#             print(self.theta_)
#             if e%1000==0:
#                 print("Epoch {}: Loss {}\n".format(e,loss))             
#                 print ('Updated Model Parameter Theta\n')
#                 print ('Sampling Next Batch of Tasks \n')
#                 print ('---------------------------------\n')
            
        total_accuracy = 0
        
        for i in range(self.num_tasks):
            for wi in range(2,len(self.windows),3):
#                 XTest, YTest = sample_points(*self.windows[wi])
                XTest, YTest = sample_points_test()
                a = np.matmul(XTest, self.theta)
                YPred = self.sigmoid(a)
                
                YPred = [self.classify(pred) for pred in YPred]
                YTest = [self.classify(test) for test in YTest]
                
                correct = 0
                for index in range(self.num_test_samples):
                    if YPred[index] == YTest[index]: correct += 1
                accuracy = (correct/self.num_test_samples) * 100
                print("Predicted {}".format(YPred))
                print("Actual {}".format(YTest))
                print("Accuracy {}%\n".format(accuracy))
                
                total_accuracy += accuracy
                
        total_accuracy = total_accuracy / (self.num_tasks * len(self.windows)/3)
        print("Total accuracy {}%".format(total_accuracy))
model = MAML()
model.train()