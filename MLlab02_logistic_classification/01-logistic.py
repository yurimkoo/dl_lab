#common linear reg's problems
#1. x 값이 커지면 linear한 line이 결과에 오류를 낼 수 있다.
#2. x 값이 커지면 1, 0이 아니라 상수가 결과로 나올 수 있다.

#logistic func. (=sigmoid func.) S shape
#features
#1. x 축의 값이 커져도 y 값이 1에 가까워질뿐 더 이상 커지진 않음.
#2. x 축의 값이 작아져도 y 값이 0에 가까워질뿐 더 이상 커지진 않음.
#H(x) = g(z) <= logistic func.
#H(x) = 1 / 1 + exp(-WX)

#cost func. for new hypothesis (apply logistic func.)
#problem: local minimize cost (구불구불 거려서 최저값에 도달하지 않아도 멈출 위험이 있음)
#C(H(x), y) = -ylog(H(x)) - (1-y)log(1-H(x))
#y=1 , C = -log(H(x))
#y=0 , C = -log(1-H(x))

#minimize cost : gradient descent algorithm

import tensorflow as tf
import numpy as np

xy = np.loadtxt('logistic_train.txt', unpack = True, dtype = 'float32')
x_data = xy[0:-1]
y_data = xy[-1]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1, len(x_data)], -1., 1.))

h = tf.matmul(W, X)
hypothesis = tf.div(1., 1. + tf.exp(-h)) #tf.div = divide 함수. 즉, 1.0 나누기 exp(-h)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

#test ML
#X의 첫번째 변수: bias, 두번째: hours, 세번째: attendance
#0.5보다 크냐고 물음으로써 결과는 true or false로 반환 (1: pass, 0:fail)
print('----------------------------')
print(sess.run(hypothesis, feed_dict={X: [[1], [2], [2]]}) > 0.5)
print(sess.run(hypothesis, feed_dict={X: [[1], [5], [5]]}) > 0.5)
print(sess.run(hypothesis, feed_dict={X: [[1, 1], [4, 3], [3, 5]]}) > 0.5) #두명을 한꺼번에 test. 첫번째 벡터: bias, 두번째: hours, 세번째: attendance
