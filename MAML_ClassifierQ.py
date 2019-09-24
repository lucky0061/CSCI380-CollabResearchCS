import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import strawberryfields as sf
from strawberryfields.ops import *
import tensorflow as tf

df = pd.read_csv('AToy.csv')
df = df.drop(df.columns[0], axis =1)

def sample_points(k, l):
    x1 = df.loc[k:l-1,['A.Open', 'A.High', 'A.Low', 'A.Close']]
    y1 = df.loc[k:l-1,['Y']]
    x2 = np.array(x1.values.tolist())
    y2 = np.array(y1.values.tolist())
    scaler0 = MinMaxScaler()
    scaler1 = MinMaxScaler()
    scaler0.fit(x2)
    scaler1.fit(y2)
    x = scaler0.transform(x2)
    y = scaler1.transform(y2)
    return x,y

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
        self.num_tasks = 1
        
        #number of samples i.e number of shots  -number of data points (k) we need to have in each task
        self.num_train_samples = 20
        self.num_meta_samples = 5
        self.num_test_samples = 5
        self.data_length = 100
        self.shift = 10

        #number of epochs i.e training iterations
        self.epochs = 3
        
        #hyperparameter for the inner loop (inner gradient update)
        self.alpha = 0.0001
        
        #hyperparameter for the outer loop (outer gradient update) i.e meta optimization
        self.beta = 0.0001
       
        #randomly initialize our model parameter theta
        self.theta = np.random.normal(size=4).reshape(4, 1)
        self.windows = mkWindows(self.num_train_samples,self.num_meta_samples,
                                 self.num_test_samples,self.data_length,self.shift)
        # Define the variational circuit and its output.
        self.X = tf.placeholder(tf.float32, shape=[4])
        self.y = tf.placeholder(tf.float32, shape=[1])
        self.sdev = 0.05
        self.depth = 80
        self.phi = tf.Variable(tf.random_normal(shape=[self.depth], stddev=self.sdev))
        self.phi_ = tf.Variable(tf.random_normal(shape=[self.depth], stddev=self.sdev))

        # eng, q = sf.Engine(3)
        self.circuit = sf.Program(4)
        self.eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": 7})
      
    #define our sigmoid activation function  
    def sigmoid(self,a):
        return 1.0 / (1 + np.exp(-a))
    
    def classify(self,value):
        return 1 if value > 0.5 else 0    
    
    #now let us get to the interesting part i.e training :P
    def train(self):
        for wi in range(0,len(self.windows),3):
            with self.circuit.context as q: #with eng:
                    Dgate(self.X[0], 0.) | q[0]
                    Dgate(self.X[1], 0.) | q[1]
                    Dgate(self.X[2], 0.) | q[2]
                    Dgate(self.X[3], 0.) | q[3]
                    #######
                    Dgate(self.phi[0], self.phi[1]) | q[0]
                    Dgate(self.phi[2], self.phi[3]) | q[1]
                    Dgate(self.phi[4], self.phi[5]) | q[2]
                    Dgate(self.phi[6], self.phi[7]) | q[3]
                    ######
                    Kgate(self.phi[24]) | q[0]
                    Kgate(self.phi[25]) | q[1]
                    Kgate(self.phi[26]) | q[2]
                    Kgate(self.phi[27]) | q[3]
                    ######
                    BSgate(self.phi[30]) | (q[0], q[1])
                    BSgate() | (q[0], q[1])
                    BSgate(self.phi[31]) | (q[1], q[2])
                    BSgate() | (q[1], q[2])
                    BSgate(self.phi[32]) | (q[2], q[3])
                    BSgate() | (q[2], q[3])
                    #######
                    Dgate(self.phi[35], self.phi[36]) | q[0]
                    Dgate(self.phi[37], self.phi[38]) | q[1]
                    Dgate(self.phi[39], self.phi[40]) | q[2]
                    Dgate(self.phi[41], self.phi[42]) | q[3]
                    ######
                    Sgate(self.phi[47], self.phi[48]) | q[0]
                    Sgate(self.phi[49], self.phi[50]) | q[1]
                    Sgate(self.phi[51], self.phi[52]) | q[2]
                    Sgate(self.phi[53], self.phi[54]) | q[3]
                    ######
                    Kgate(self.phi[59]) | q[0]
                    Kgate(self.phi[60]) | q[1]
                    Kgate(self.phi[61]) | q[2]
                    Kgate(self.phi[62]) | q[3]
                    #######
                    BSgate(self.phi[65]) | (q[0], q[1])
                    BSgate() | (q[0], q[1])
                    BSgate(self.phi[66]) | (q[1], q[2])
                    BSgate() | (q[1], q[2])
                    BSgate(self.phi[67]) | (q[2], q[3])
                    BSgate() | (q[2], q[3])    
            results = self.eng.run(self.circuit, run_options={"eval": False})
            # Define the output as the probability of measuring |0,2> as opposed to |2,0>
            # p0 = state.fock_prob([0, 0, 2])
            # p1 = state.fock_prob([0, 2, 0])
            # p2 = state.fock_prob([2, 0, 0])
            mean_x_0, svd_x = results.state.quad_expectation(0)
            mean_x_1, svd_x = results.state.quad_expectation(1)
            mean_x_2, svd_x = results.state.quad_expectation(2)
            mean_x_3, svd_x = results.state.quad_expectation(3)
            # mean_x_4, svd_x = results.state.quad_expectation(4)
            # mean_x_5, svd_x = results.state.quad_expectation(5)
            # mean_x_2, svd_x = results.state.quad_expectation(2)
            # norm = mean_x_0 + mean_x_1 + mean_x_2 + mean_x_3 + mean_x_4 + mean_x_5 + 1e-10
            norm = mean_x_0 + mean_x_1 + mean_x_2 + mean_x_3 + 1e-10
            circuit_output = [mean_x_0/norm]
            # circuit_output = [mean_x_0, mean_x_1, mean_x_2]

            loss = tf.losses.mean_squared_error(labels=circuit_output, predictions=self.y)
            #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = circuit_output))
            optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
            minimize_op = optimizer.minimize(loss)
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            output = tf.round(circuit_output)
            #for the number of epochs,
            for e in range(self.epochs):    

                self.theta_ = []

                #for task i in batch of tasks
                for i in range(self.num_tasks):

                    #sample k data points and prepare our train set
                    XTrain, YTrain = sample_points(*self.windows[wi])
    #                 XTrain = X_train[:20]
    #                 YTrain = Y_train[:20]
                    for j in range(len(YTrain)):
                        sess.run([minimize_op], feed_dict={self.X: XTrain[j], self.y: YTrain[j]})

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
                    XMeta, YMeta = sample_points(*self.windows[wi+1])
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
            
        for i in range(self.num_tasks):
            for wi in range(2,len(self.windows),3):
                XTest, YTest = sample_points(*self.windows[wi])
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
model = MAML()
model.train()