import tensorflow as tf
import numpy as np

xy = np.loadtxt('XOR_train.txt', unpack=True)
x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1], (4, 1))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2, 2], -1., 1.)) #처음 weight 은 2x2 matrix
W2 = tf.Variable(tf.random_uniform([2, 1], -1., 1.)) #두번째 weight 은 2x1 matrix

b1 = tf.Variable(tf.zeros([2]), name = 'Bias1') #처음 bias 는 2x1 matrix 로 값이 2개
b2 = tf.Variable(tf.zeros([1]), name = 'Bias2') #두번째 bias 는 값이 1개

L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
hypothesis = tf.sigmoid(tf.matmul(L2, W2) + b2)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))

optimizer = tf.train.GradientDescentOptimizer(0.3).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

    #test model
    correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)

    #calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    print(sess.run([hypothesis, tf.floor(hypothesis + 0.5), correct_prediction, accuracy], feed_dict={X: x_data, Y: y_data}))
    print('Accuracy:', accuracy.eval({X: x_data, Y: y_data}))
