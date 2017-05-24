#learning rate
#overshooting : 너무 큰 step(learning rate)을 가질 때 convex func. 밖으로 값이 튕겨나가는 현상
#small learning rate: 너무 작은 learning rate 을 가질 때 최저점이 아님에도 불구하고 stop
#learning rate 이 의심될 때 cost func.을 출력해본다 (default = 0.01)

#data preprocessing for gradient descent : multi-variable 에서 n개의 x값이 차이가 심할 때 발산할 위험.
#해결 => 1)zero-centered data -> 2)normalize 해야함.
#standardization : Xj' = Xj - mean(j) / std(j)

#overfitting : training data set 에 너무 잘 맞는 model
#해결 => 1)more training data 2)reduce the number of features 3)regularization(일반화)
#regularization : weight 의 값을 너무 크게 갖지 말자. (데이터에 맞게 구부리지 말고 펴자!)

#evaluation ML model
#70% = training data set / 30% = test data set
#training set을 가지고 ML 시킨 후 validation set을 가지고 alpha, lambda의 값을 튜닝, test set을 가지고 evaluation model

#online learning
#100만개의 데이터를 10만개씩 나눠서 각각 학습시키고, 나중에 추가 데이터만 넣어서 학습시키면 됨

import tensorflow as tf
import numpy as np

xy = np.loadtxt('softmax_train.txt', unpack=True, dtype='float32')
x_data = np.transpose(xy[0:3])
y_data = np.transpose(xy[3:])

X = tf.placeholder('float', [None, 3])
Y = tf.placeholder('float', [None, 3])

W = tf.Variable(tf.zeros([3, 3]))

hypothesis = tf.nn.softmax(tf.matmul(X, W))

#learning rate을 10.으로 큰 값을 줄 경우, 결과에 처음에는 어느 정도 값이 나오다가 이후에는 'nan'이 나온다. (너무 커서 안 나오는 것)
#learning rate을 0.001으로 할 경우, 값이 너무 작아 cost 가 미세하게 줄어듦
#계속된 시행착오를 통해 model에 잘 맞는 learning rate을 찾는 게 중요하다.
learning_rate = 0.1

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), reduction_indices=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))


