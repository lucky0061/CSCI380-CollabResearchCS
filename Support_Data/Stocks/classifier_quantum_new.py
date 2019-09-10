import strawberryfields as sf
from strawberryfields.ops import *
import tensorflow as tf

# Define the variational circuit and its output.
X = tf.placeholder(tf.float32, shape=[6])
y = tf.placeholder(tf.float32, shape=[1])
sdev = 0.05
depth = 70
phi = tf.Variable(tf.random_normal(shape=[depth], stddev=sdev))

# eng, q = sf.Engine(3)
eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": 10})
circuit = sf.Program(6)

with circuit.context as q: #with eng:
	# Note that we are feeding 1-d tensors into gates, not scalars!
	Dgate(X[0], 0.) | q[0]
	Dgate(X[1], 0.) | q[1]
	Dgate(X[2], 0.) | q[2]
	Dgate(X[3], 0.) | q[3]
	Dgate(X[4], 0.) | q[4]
	Dgate(X[5], 0.) | q[5]
	# Dgate(X[0], X[1]) | q[2]
	Dgate(phi[0], phi[1]) | q[0]
	Dgate(phi[2], phi[3]) | q[1]
	Dgate(phi[4], phi[5]) | q[2]
	Dgate(phi[6], phi[7]) | q[3]
	Dgate(phi[8], phi[9]) | q[4]
	Dgate(phi[10], phi[11]) | q[5]

	# Dgate(phi[4], phi[5]) | q[2]
	Sgate(phi[12], phi[13]) | q[0]
	Sgate(phi[14], phi[15]) | q[1]
	Sgate(phi[16], phi[17]) | q[2]
	Sgate(phi[18], phi[19]) | q[3]
	Sgate(phi[20], phi[21]) | q[4]
	Sgate(phi[22], phi[23]) | q[5]
	# Sgate(phi[10], phi[11]) | q[2]
	Kgate(phi[24]) | q[0]
	Kgate(phi[25]) | q[1]
	Kgate(phi[26]) | q[2]
	Kgate(phi[27]) | q[3]
	Kgate(phi[28]) | q[4]
	Kgate(phi[29]) | q[5]
	# Kgate(phi[32]) | q[2]
	BSgate(phi[30]) | (q[0], q[1])
	BSgate() | (q[0], q[1])
	BSgate(phi[31]) | (q[1], q[2])
	BSgate() | (q[1], q[2])
	BSgate(phi[32]) | (q[2], q[3])
	BSgate() | (q[2], q[3])
	BSgate(phi[33]) | (q[3], q[4])
	BSgate() | (q[3], q[4])
	BSgate(phi[34]) | (q[4], q[5])
	BSgate() | (q[4], q[5])
	# BSgate(phi[13]) | (q[0], q[2])
	# BSgate() | (q[0], q[2])
	# BSgate(phi[14]) | (q[1], q[2])
	# BSgate() | (q[1], q[2])
	
	Dgate(phi[35], phi[36]) | q[0]
	Dgate(phi[37], phi[38]) | q[1]
	Dgate(phi[39], phi[40]) | q[2]
	Dgate(phi[41], phi[42]) | q[3]
	Dgate(phi[43], phi[44]) | q[4]
	Dgate(phi[45], phi[46]) | q[5]
	# Dgate(phi[19], phi[20]) | q[2]
	Sgate(phi[47], phi[48]) | q[0]
	Sgate(phi[49], phi[50]) | q[1]
	Sgate(phi[51], phi[52]) | q[2]
	Sgate(phi[53], phi[54]) | q[3]
	Sgate(phi[55], phi[56]) | q[4]
	Sgate(phi[57], phi[58]) | q[5]
	# Sgate(phi[25], phi[26]) | q[2]
	Kgate(phi[59]) | q[0]
	Kgate(phi[60]) | q[1]
	Kgate(phi[61]) | q[2]
	Kgate(phi[62]) | q[3]
	Kgate(phi[63]) | q[4]
	Kgate(phi[64]) | q[5]
	# Kgate(phi[35]) | q[2]
	BSgate(phi[65]) | (q[0], q[1])
	BSgate() | (q[0], q[1])
	BSgate(phi[66]) | (q[1], q[2])
	BSgate() | (q[1], q[2])
	BSgate(phi[67]) | (q[2], q[3])
	BSgate() | (q[2], q[3])
	BSgate(phi[68]) | (q[3], q[4])
	BSgate() | (q[3], q[4])
	BSgate(phi[69]) | (q[4], q[5])
	BSgate() | (q[4], q[5])
	# BSgate(phi[28]) | (q[0], q[2])
	# BSgate() | (q[0], q[2])
	# BSgate(phi[29]) | (q[1], q[2])
	# BSgate() | (q[1], q[2])


# state = eng.run('tf', cutoff_dim=10, eval=False)
results = eng.run(circuit, run_options={"eval": False})
# Define the output as the probability of measuring |0,2> as opposed to |2,0>
# p0 = state.fock_prob([0, 0, 2])
# p1 = state.fock_prob([0, 2, 0])
# p2 = state.fock_prob([2, 0, 0])
mean_x_0, svd_x = results.state.quad_expectation(0)
mean_x_1, svd_x = results.state.quad_expectation(1)
# mean_x_2, svd_x = results.state.quad_expectation(2)
circuit_output = [mean_x_0, mean_x_1]
# circuit_output = [mean_x_0, mean_x_1, mean_x_2]

loss = tf.losses.mean_squared_error(labels=circuit_output, predictions=y)
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = circuit_output))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
minimize_op = optimizer.minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
output = tf.round(circuit_output)

# Generate some data
X_train = [[0.2, 0.4], [0.6, 0.8], [0.4, 0.2], [0.8, 0.6]]
Y_train = [[0, 1], [0, 1], [1, 0], [1, 0]]

steps = 100

for i in range(steps):
    # if i % 10 == 0:
        # print("Epoch {0}, Loss {1}".format(i, sess.run([loss], feed_dict={X: X_train[0], y: Y_train[0]})[0]))
    for j in range(len(Y_train)):
        sess.run([minimize_op], feed_dict={X: X_train[j], y: Y_train[j]})
    

print("X       Prediction       Label")
for i in range(4):
    print("{0} || {1} || {2}".format(X_train[i], sess.run(output, feed_dict={X: X_train[i]}), Y_train[i]))