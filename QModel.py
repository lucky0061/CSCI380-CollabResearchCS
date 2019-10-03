import numpy as np
import random
import os
from collections import deque
import strawberryfields as sf

from strawberryfields.ops import *
import tensorflow as tf 
class QModel:
    def __init__(self, session,theta = tf.Variable([0.1]*3) ):
        self.x = tf.placeholder(tf.float32, shape = [1])
        self.y = tf.placeholder(tf.float32, shape = [1])
        self.para = theta
        eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": 7})
        circuit = sf.Program(1)

        with circuit.context as q: 
            Dgate(self.x[0]) | q[0]

            Dgate(self.para[0]) | q[0]

            Sgate(self.para[1]) | q[0]

            Vgate(self.para[2]) | q[0]

        results = eng.run(circuit, run_options={"eval": False})
        output, _ = results.state.quad_expectation(0)
        self.output = tf.Variable(0.0)
        self.output = tf.assign(self.output,tf.add(self.output, output))
        # self.output = tf.output
       
        self.loss = tf.losses.mean_squared_error(labels=[output], predictions=self.y)
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.y, logits = [output]))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

        self.train = optimizer.minimize(self.loss)
        self.sess = session
        self.sess.run(tf.global_variables_initializer())

        # self.Shape = x_size

    def calc_grad(self, X, Y):
        # gradient = tf.placeholder(tf.float32, shape = [self.Shape,1])
        gradient = []
        grad = tf.gradients(self.loss, self.para)[0]
        # for i in range(len(X)):
        # print("train ", self.train,  " ------------------- ")
        gradient.append(self.sess.run(grad, {self.x: X[-1], self.y: Y[-1]}))
        return np.array(gradient)
        
    def fit(self,X, Y):
        # grad,_ = self.sess.run(self.train, {self.x: X_j, self.y: Y_j})
        for i in range(len(X)):
            self.sess.run(self.train, {self.x: X[i], self.y: Y[i]})
        # return tf.reshape(grad, [3,1])
    # def fit(X_j, Y_j, sess):
        # sess.run(train, {x: [X_j], y: [Y_j]})
        # return sess.run(cross_entropy, {x: [X_j], y: [Y_j]})
    def predict(self,X):
        pred = []
        for i in range(len(X)):
	        pred += self.sess.run(self.output,{self.x: X[i]})
        print("pred: ", pred, " +++++++++++++++ ")
        return pred
        
# class QMetaOptimizer:
#     def __init__(self, session, theta_):
#         self.x = tf.placeholder(tf.float32, shape=[1])
#         self.y = tf.placeholder(tf.float32, shape=[1])
#         self.para = tf.Variable(theta_)
#         eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": 7})
#         circuit = sf.Program(1)

#         with circuit.context as q: 
#             Dgate(self.x[0]) | q[0]

#             Dgate(self.para[0]) | q[0]

#             Sgate(self.para[1]) | q[0]

#             Vgate(self.para[2]) | q[0]

#         results = eng.run(circuit, run_options={"eval": False})
#         output, _ = results.state.quad_expectation(0)
       
#         self.loss = tf.losses.mean_squared_error(labels=[output], predictions=self.y)
#         # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.y, logits = [output]))
#         optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

#         self.train = optimizer.minimize(self.loss)
#         self.sess = session
#         self.sess.run(tf.global_variables_initializer())
#         # self.Shape = x_size
#         def calc_grad(self, X, Y):
#             # gradient = tf.placeholder(tf.float32, shape = [self.Shape,1])
#             gradient = []
#             grad = tf.gradients(self.loss, self.para)[0]
#             # for i in range(len(X)):
#             # print("train ", self.train,  " ------------------- ")
#             gradient.append(self.sess.run(grad, {self.x: X[len(X) -1], self.y: Y[len(X)-1]}))
#             return np.array(gradient)
#         def fit(self,X, Y):
#             # self.sess.run(tf.global_variables_initializer())
#             for i in range(len(X)):
#                 self.sess.run(train, {self.x: X[i], self.y: Y[i]})
#                 return self.sess.run(self.para)
#         # # eng, q = sf.Engine(3)
#         # eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": 7})
#         # circuit = sf.Program(3)

#         # with circuit.context as q: #with eng:
#         #     Dgate(self.X[0], 0.) | q[0]
#         #     Dgate(self.X[1], 0.) | q[1]
#         #     Dgate(self.X[2], 0.) | q[2]
#         # ##	Dgate(X[4], 0.) | q[4]
#         # ##	Dgate(X[5], 0.) | q[5]
#         #     # Dgate(X[0], X[1]) | q[2]
#         #     Dgate(self.para[0]) | q[0]
#         #     Dgate(self.para[1]) | q[1]
#         #     Dgate(self.para[2]) | q[2]
#         # ##	Dgate(para[8], para[9]) | q[4]
#         # ##	Dgate(para[10], para[11]) | q[5]

#         #     # Dgate(para[4], para[5]) | q[2]
#         #     Sgate(self.para[3]) | q[0]
#         #     Sgate(self.para[4]) | q[1]
#         #     Sgate(self.para[5]) | q[2]
#         # ##	Sgate(para[20], para[21]) | q[4]
#         # ##	Sgate(para[22], para[23]) | q[5]
#         #     # Sgate(para[10], para[11]) | q[2]
#         #     Kgate(self.para[6]) | q[0]
#         #     Kgate(self.para[7]) | q[1]
#         #     Kgate(self.para[8]) | q[2]
#         #     BSgate(self.para[9]) | (q[0], q[1])
#         #     BSgate() | (q[0], q[1])
#         #     BSgate(self.para[10]) | (q[1], q[2])
#         #     BSgate() | (q[1], q[2])
            
#         # results = eng.run(circuit, run_options={"eval": False})
#         # output_0, _ = results.state.quad_expectation(0)
#         # output_1, _ = results.state.quad_expectation(1)
#         # output_2, _ = results.state.quad_expectation(2)
#         # norm = output_0 + output_1 + output_2+ 1e-10
#         # circuit_output = [output_0/norm]
#         # loss = tf.losses.mean_squared_error(labels=circuit_output, predictions=y)
#         # optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
#         # train = optimizer.minimize(loss)
#         # self.sess = session
