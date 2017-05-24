#XOR: 둘이 같으면 0, 둘이 다르면 1

#NN(feature이 두개일때) //forward propagation
#K(x)=sigmoid(XW1+B1)
#H(x)=sigmoid(K(x)W2+b2)

#NN // back-propagation
#chain rule 적용 in derivation(미분)
#K = tf.sigmoid(tf.matmul(X, W1) + b1) 과 같이 sigmoid 적용
#hypothesis = tf.sigmoid(tf.matmul(K, W2) + b2)

import tensorflow as tf
import numpy as np

xy = np.loadtxt('XOR_train.txt', unpack=True)
x_data = xy[0:-1]
y_data = xy[-1]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1, len(x_data)], -1., 1.))

h = tf.matmul(W, X)
hypothesis = tf.div(1., 1 + tf.exp(-h))

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))

a = tf.Variable(0.01)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(1000):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

    #test model
    correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)

    #calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    print(sess.run([hypothesis, tf.floor(hypothesis + 0.5), correct_prediction, accuracy], feed_dict={X: x_data, Y: y_data})) #floor: 버림 함수
    print('Accuracy:', accuracy.eval({X: x_data, Y: y_data}))