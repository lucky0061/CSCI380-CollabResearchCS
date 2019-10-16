import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import strawberryfields as sf
from strawberryfields.ops import *
from QModel import QModel



df0 = pd.read_csv('A.csv')
df0 = df0.drop(df0.columns[0], axis =1)
df1 = pd.read_csv('DVA.csv')
df1 = df1.drop(df1.columns[0], axis =1)
df2 = pd.read_csv('XRAY.csv')
df2 = df2.drop(df2.columns[0], axis =1)
df3 = pd.read_csv('GRMN.csv')
df3 = df3.drop(df3.columns[0], axis =1)
df4 = pd.read_csv('MAC.csv')
df4 = df4.drop(df4.columns[0], axis =1)

dfs = [df0,df1,df2,df3,df4]

df5 = pd.read_csv('XEC.csv')
df5 = df5.drop(df5.columns[0], axis = 1)

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
def sample_points(df,size):
    start = random.randint(0,len(df)-size)
    end = start + size
    x1 = df.iloc[start:end,[3]]
    x2 = df.iloc[start:end,[5]]
    y1 = df.iloc[start:end,[8]]
    x1 = np.array(x1.values.tolist())
    y1 = np.array(y1.values.tolist())
    x2 = np.array(x2.values.tolist())
    scaler0 = MinMaxScaler()
    scaler0.fit(x1)
    scaler1 = MinMaxScaler()
    scaler1.fit(x2)
    # x = [scaler0.transform(x1).reshape(size,1),scaler1.transform(x2).reshape(size,1)]
    # x = [scaler0.transform(x1),scaler1.transform(x2)]
    # [list(a) for a in zip([1,2,3], [4,5,6], [7,8,9])]
    x = np.array([a for a in zip(scaler0.transform(x1),scaler1.transform(x2))])
    y = y1.reshape(size,1)
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
    def __init__(self, session):
        
        #initialize number of tasks i.e number of tasks we need in each batch of tasks
        self.num_tasks = 5
        
        #number of samples i.e number of shots  -number of data points (k) we need to have in each task
        self.num_train_samples = 5
        self.num_meta_samples = 5
        self.num_test_samples = 5
        self.data_length = 20
        self.shift = 5

        #number of epochs i.e training iterations
        self.epochs = 10
        
        #hyperparameter for the inner loop (inner gradient update)
        # self.alpha = 0.0001
        self.alpha = 0.001
        
        #hyperparameter for the outer loop (outer gradient update) i.e meta optimization
        # self.beta = 0.0001
        self.beta = 0.001
       
        #randomly initialize our model parameter theta
        # self.theta = np.random.normal(size=1).reshape(1, 1)
        self.sdev = 0.05
        self.depth = 80
        # self.theta = tf.Variable(tf.random_normal(shape=[1,1], stddev=self.sdev))
        # self.theta = np.random.normal(size=3).reshape(3)
        self.theta = np.random.normal(size=7) # tf.Variable([0.1]*9) 
        
        # self.windows = mkWindows(self.num_train_samples,self.num_meta_samples,
        #                          self.num_test_samples,self.data_length,self.shift)
                # Define the variational circuit and its output.
        # self.X = tf.placeholder(tf.float32, shape=[1])
        # self.y = tf.placeholder(tf.float32, shape=[1])
        
        # self.phi = tf.Variable(tf.random_normal(shape=[self.depth], stddev=self.sdev))
        # self.phi_ = tf.Variable(tf.random_normal(shape=[self.depth], stddev=self.sdev))

        # eng, q = sf.Engine(3)
        self.MySess = session
        # self.TrainModel = QModel(self.MySess)       # make new one every time     
        # self.MetaModel = QMetaOptimizer(session1)
      
    #define our sigmoid activation function  
    def sigmoid(self,a):
        return 1.0 / (1 + np.exp(-a))
    
    def classify(self,value):
        return 1 if value > 0.5 else 0    
    
    #now let us get to the interesting part i.e training :P
    def train(self):
        #for the number of epochs,
        for e in range(self.epochs):        
            print("Epoch: ", e+1)
            theta_ = []
            # print("Training Now", "#" * e, end = "\r")
            print("Training Now ===========================================================")
            

            #for task i in batch of tasks
            for i in range(self.num_tasks):
                TrainModel = QModel(self.MySess, self.theta)
               

                #sample k data points and prepare our train set
                XTrain, YTrain = sample_points(dfs[i],20)
                # print("SHAPE XTrain:", XTrain.shape)
                # print("XTrain: " + str(XTrain.tolist()))
                # print("YTrain: " + str(YTrain.tolist()))
                # print(self.theta)
                TrainModel.fit(XTrain,YTrain)
                gradient = TrainModel.calc_grad(XTrain,YTrain)
                print(gradient)

                
#                 a = np.matmul(XTrain, self.theta)
                
# #                 print(a)

#                 YHat = self.sigmoid(a)
# #                 print(YHat)

                #since we are performing classification, we use cross entropy loss as our loss function
#                 loss = ((np.matmul(-YTrain.T, np.log(YHat)) - np.matmul((1 -YTrain.T), np.log(1 - YHat)))/self.num_train_samples)[0][0]
# #                 print(loss)
                #minimize the loss by calculating gradients
                # gradient = np.matmul(XTrain.T, (YHat - YTrain)) / self.num_train_samples

                #update the gradients and find the optimal parameter theta' for each of tasks
                theta_.append(self.theta - self.alpha*gradient)


            #initialize meta gradients
            meta_gradient = np.zeros(self.theta.shape)

            for i in range(self.num_tasks):

                #sample k data points and prepare our test set for meta training
                XMeta, YMeta = sample_points(dfs[i],20)
#                 XTest = X_train[20:23]
#                 YTest = Y_train[20:23]
                # Meta =  QMetaOptimizer(session1, theta_[i])
                # temp = Meta.fit(XMeta,YMeta)
                # print("Theta_[i]: ", theta_[i], " -------------------- ")
                # print("Theta_[i][0]: ", theta_[i][0], " -------------------- ")
                print("Meta Now --------------------------------------------- ")
                QMetaModel = QModel(self.MySess,tf.Variable(theta_[i][0]))
                QMetaModel.fit(XMeta, YMeta)
                meta_gradient += QMetaModel.calc_grad(XMeta, YMeta)[0]

                # #predict the value of y
                # a = np.matmul(XMeta, self.theta_[i])

                # YPred = self.sigmoid(a)

                # #compute meta gradients
                # meta_gradient += np.matmul(XMeta.T, (YPred - YMeta)) / self.num_meta_samples


            #update our randomly initialized model parameter theta with the meta gradients
            self.theta = self.theta-self.beta*meta_gradient/self.num_tasks
    def test(self,df, SampleSize, TestSize):     
        total_accuracy = 0
        
        TestModel = QModel(self.MySess, self.theta)
        # for i in range(self.num_tasks):
        train_start = random.randint(0,SampleSize)
        train_end = train_start + SampleSize
        test_end = train_end + TestSize
        XTrain, YTrain = sample_points(df,SampleSize)
        TestModel.fit(XTrain,YTrain)
        print("Testing Now +_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_")
        XTest, YTest = sample_points(df, TestSize)
        # a = np.matmul(XTest, self.theta)
        # YPred = self.sigmoid(a)
        YPred = TestModel.predict(XTest)
        YPred = [self.classify(pred) for pred in YPred]
        # print("YPred: ", YPred, " --------------------- ")
        # YTest = [self.classify(test) for test in YTest]
        
        correct = 0
        for index in range(0,TestSize):
            if YPred[index] == YTest[index]: correct += 1
        accuracy = (correct/self.num_test_samples) * 100
        print("Predicted {}".format(YPred))
        print("Actual {}".format(YTest))
        print("Accuracy {}%\n".format(accuracy))
        
        total_accuracy += accuracy

        # total_accuracy = total_accuracy / self.num_tasks
        print("Total accuracy {}%".format(total_accuracy))
        
model = MAML(tf.Session())
model.train()
model.test(df5, 20, 5)