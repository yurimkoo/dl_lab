#simplified hypothesis
#H(x) = Wx
#cost(W) = sum(n)(Wx^(n)-y^(n))^2 / n
#↓
#gradient descent algorithm = minimize algorithm (multi values도 가능)
#경사도를 따라 내려가면서 최저점에 도달하는 방법
#경사도를 구하는 법: 미분
#linear algorithm을 이용할 때 convex function이 되어야 좋음 (아니면 엉뚱한 곳으로 값을 찾아감)

import tensorflow as tf
import matplotlib.pyplot as plt

X = [1., 2., 3.]
Y = [1., 2., 3.]
m = n_samples = len(X)

W = tf.placeholder(tf.float32)

hypothesis = tf.mul(X, W)

cost = tf.reduce_sum(tf.pow(hypothesis-Y, 2))/(m)

init = tf.initialize_all_variables()

W_val = []
cost_val = []

sess = tf.Session()
sess.run(init)
#-30에서 50까지 0.1 단위로 반복
for i in range(-30, 50):
    print(i*0.1, sess.run(cost, feed_dict={W: i*0.1}))
    W_val.append(i*0.1)
    cost_val.append(sess.run(cost, feed_dict={W: i*0.1}))

plt.plot(W_val, cost_val, 'ro')
plt.ylabel('Cost')
plt.xlabel('W')
plt.show()
