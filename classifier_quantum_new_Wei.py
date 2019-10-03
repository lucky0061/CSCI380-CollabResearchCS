import strawberryfields as sf
from strawberryfields.ops import *
import tensorflow as tf

# Define the variational circuit and its output.
X = tf.placeholder(tf.float32, shape=[2])
y = tf.placeholder(tf.float32, shape=[2])
sdev = 0.05
depth = 50
phi = tf.Variable(tf.random_normal(shape=[depth], stddev=sdev))

# eng, q = sf.Engine(3)
eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": 10})
circuit = sf.Program(2)

with circuit.context as q: #with eng:
	# Note that we are feeding 1-d tensors into gates, not scalars!
	Dgate(X[0], 0.) | q[0]
	Dgate(X[1], 0.) | q[1]
	# Dgate(X[0], X[1]) | q[2]
	Dgate(phi[0], phi[1]) | q[0]
	Dgate(phi[2], phi[3]) | q[1]
	# Dgate(phi[4], phi[5]) | q[2]
	Sgate(phi[6], phi[7]) | q[0]
	Sgate(phi[8], phi[9]) | q[1]
	# Sgate(phi[10], phi[11]) | q[2]
	Kgate(phi[30]) | q[0]
	Kgate(phi[31]) | q[1]
	# Kgate(phi[32]) | q[2]
	BSgate(phi[12]) | (q[0], q[1])
	BSgate() | (q[0], q[1])
	# BSgate(phi[13]) | (q[0], q[2])
	# BSgate() | (q[0], q[2])
	# BSgate(phi[14]) | (q[1], q[2])
	# BSgate() | (q[1], q[2])
	
	Dgate(phi[15], phi[16]) | q[0]
	Dgate(phi[17], phi[18]) | q[1]
	# Dgate(phi[19], phi[20]) | q[2]
	Sgate(phi[21], phi[22]) | q[0]
	Sgate(phi[23], phi[24]) | q[1]
	# Sgate(phi[25], phi[26]) | q[2]
	Kgate(phi[33]) | q[0]
	Kgate(phi[34]) | q[1]
	# Kgate(phi[35]) | q[2]
	BSgate(phi[27]) | (q[0], q[1])
	BSgate() | (q[0], q[1])
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