#linear regression

#cost func. (=Loss func.) : 가설과 data의 거리를 구하는 함수 //  (가장 작은 값을 구하는 것)
#H(x) = Wx + b , cost(W, b) = sum(n)(H(x^(n))-y^(n))^2 / n
#goal : minimize cost

import tensorflow as tf

#train data
x_data = [1,2,3]
y_data = [1,2,3]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#H(x) = Wx + b
hypothesis = W * X + b

#cost(W,b) = sum(n)(H(x^(n))-y^(n))^2
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#blackbox
a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

#변수 초기화 (에러메시지 방지)
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

#각 스텝 출력 (2000번 시행, 20번 마다 시행)
for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X: x_data, Y:y_data}), sess.run(W), sess.run(b))

#model 이용 predict values
print(sess.run(hypothesis, feed_dict={X:5}))
print(sess.run(hypothesis, feed_dict={X:2.5}))