#Vanishing Gradient 문제 해결
#1. ReLU
#2. initialize weights

#initialize weights
#if we set all initial weights to 0 : 모든 값이 0이 되어 좋지 않음 (절대 안 됨)
#RBM (restricted boltzmann machine) // Deep Belief Network (pre-training , fine tuning)

#simple methods are OK
#Xavier initialization : RBM보다 단순하고 효과는 비슷

import tensorflow as tf
import matplotlib.pyplot as plt
import random
from tensorflow.examples.tutorials.mnist import input_data

def xavier_init(n_inputs, n_outputs, uniform=True):
    '''n_inputs: The number of input nodes into each output
       n_outputs: The number of output nodes for each input
       uniform: If true use a uniform distribution, otherwise use a normal

       returns: an initializer'''
    if uniform:
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)

learning_rate = 0.001
training_epoch = 15
display_step = 1
batch_size = 100

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

X = tf.placeholder('float', [None, 784])
Y = tf.placeholder('float', [None, 10])

W1 = tf.get_variable('W1', shape=[784, 256], initializer=xavier_init(784, 256))
W2 = tf.get_variable('W2', shape=[256, 256], initializer=xavier_init(256, 256))
W3 = tf.get_variable('W3', shape=[256, 10], initializer=xavier_init(256, 10))

b1 = tf.Variable(tf.zeros([256]))
b2 = tf.Variable(tf.zeros([256]))
b3 = tf.Variable(tf.zeros([10]))

L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))
L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), b2))

hypothesis = tf.add(tf.matmul(L2, W3), b3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epoch):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += sess.run(cost, feed_dict={X: batch_xs, Y: batch_ys}) / total_batch
        if epoch % display_step == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost))

    print('Optimization Finished!')

    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    print('Accuracy:', accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))

    # predict & show
    r = random.randint(0, mnist.test.num_examples - 1)
    print('Label:', sess.run(tf.argmax(hypothesis, 1), {X: mnist.test.images[r:r + 1]}))
    print('Prediction:', sess.run(tf.argmax(hypothesis, 1), {X: mnist.test.images[r:r + 1]}))

    # show the img
    plt.imshow(mnist.test.images[r:r + 1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()