import tensorflow as tf
import numpy as np

xy = np.loadtxt('C:\\Users\\asus\\Desktop\\train.txt', unpack = 1, dtype = 'float32')
x_data = xy[0:-1]
y_data = xy[-1]

print('x', x_data)
print('y', y_data)

W = tf.Variable(tf.random_uniform([1, len(x_data)], -5., 5.))

hypothesis = tf.matmul(W, x_data)

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W))