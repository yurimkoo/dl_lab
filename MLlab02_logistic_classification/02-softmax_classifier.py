#softmax
#1. 0~1의 값이다
#2. 전체성이 있다 (값이 확률이다)

#one-hot encoding : n 개 중 제일 큰 하나만 골라서 1.0 로 만들고 나머지는 다 0으로 만듦 (in tf: argmax)

#D(S, L) = - sum(i) (Li * log(Si)) // S: H(x), L: y

import tensorflow as tf
import numpy as np

#'soft_max.txt' uses 'one-hot encoding'
xy = np.loadtxt('softmax_train.txt', unpack=1, dtype='float32')
x_data = np.transpose(xy[0:3]) #transpose를 사용하는 것은 거울처럼 반대로 뒤집어달라는 뜻
y_data = np.transpose(xy[3:])

X = tf.placeholder('float', [None, 3]) #x1, x2, bias(x0) total 3 (None인 이유는 총 데이터 개수가 몇 개인지 모르기 때문)
Y = tf.placeholder('float', [None, 3]) #A, B, C total 3

W = tf.Variable(tf.zeros([3, 3])) #3x3 matrix (첫번째 값: x가 3개, 두번째 값: y가 3개) // tf.zeros 는 행렬 만드는 함수

hypothesis = tf.nn.softmax(tf.matmul(X, W)) #뒤집는 이유는 계산상의 편의를 위해해
learning_rate = 0.001

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), reduction_indices=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 20 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

#test & one-hot encoding
a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7]]})
print(a, sess.run(tf.arg_max(a, 1)))

b = sess.run(hypothesis, feed_dict={X: [[1, 3, 4]]})
print(b, sess.run(tf.arg_max(b, 1)))

c = sess.run(hypothesis, feed_dict={X: [[1, 1, 0]]})
print(c, sess.run(tf.arg_max(c, 1)))

all = sess.run(hypothesis, feed_dict={X: [[1, 11, 7], [1, 3, 4], [1, 1, 0]]})
print(all, sess.run(tf.arg_max(all, 1)))

