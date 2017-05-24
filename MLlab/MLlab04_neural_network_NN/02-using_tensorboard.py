#tensorflow summary 관련 함수 표기법 수정

import tensorflow as tf
import numpy as np

xy = np.loadtxt('XOR_train.txt', unpack=True)
x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1], (4, 1))

X = tf.placeholder(tf.float32, name = 'X-input')
Y = tf.placeholder(tf.float32, name = 'Y-input')

W1 = tf.Variable(tf.random_uniform([2, 2], -1., 1.), name = 'Weight1')
W2 = tf.Variable(tf.random_uniform([2, 1], -1., 1.), name = 'Weight2')

b1 = tf.Variable(tf.zeros([2]), name = 'Bias1')
b2 = tf.Variable(tf.zeros([1]), name = 'Bias2')

#Grouping

with tf.name_scope('Layer2') as scope:
    L2 = tf.sigmoid(tf.matmul(X, W1) + b1)

with tf.name_scope('Layer3') as scope:
    hypothesis = tf.sigmoid(tf.matmul(L2, W2) + b2)

with tf.name_scope('Cost') as scope:
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))
    cost_sum = tf.summary.scalar('Cost', cost) #scalar variable : 하나의 값 (벡터가 아닌)

with tf.name_scope('train') as scope:
    optimizer = tf.train.GradientDescentOptimizer(0.3).minimize(cost)

with tf.name_scope('Accuracy') as scope:
    correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    accuracy_sum = tf.summary.scalar('Accuracy', accuracy)

#Add histogram

w1_hist = tf.summary.histogram('Weight1', W1)
w2_hist = tf.summary.histogram('Weight2', W2)

b1_hist = tf.summary.histogram('Bias1', b1)
b2_hist = tf.summary.histogram('Bias2', b2)

y_hist = tf.summary.histogram('y', Y)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./log/xor_logs', sess.graph)

    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            summary = sess.run(merged, feed_dict={X: x_data, Y: y_data})
            writer.add_summary(summary, step)

    # test model
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))

    # calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    print('Accuracy:', accuracy.eval({X: x_data, Y: y_data}))
