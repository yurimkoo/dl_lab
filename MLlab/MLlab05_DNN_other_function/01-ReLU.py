#sigmoid의 단점: chain rule을 통해 계속 미분해서 곱해져도 작은 값이 유지됨 (0에 가까운 값)
#그래서 hidden layers가 많아져도 accuracy가 낮아지고 cost는 높아짐

#ReLU: sigmoid를 linear하게 표현. x값이 0보다 작을 때는 0으로 유지되지만, 0보다 클 경우 1까지가 아닌 계속 증가할 수 있도록.
#Hidden layer에서는 Rectified Linear Unit (ReLU) 사용 -> sigmoid를 NN에서 사용하는 것 X
#BUT 마지막 output layer는 sigmoid를 사용한다.

#ReLU의 다른 형태
#Leaky ReLU : 0보다 작은 값에 대해 0.1을 곱함
#ELU : 0.1이 아니고 다른 값을 곱해보자

#sigmoid의 다른 형태
#tanh : 0을 기준으로 -1 ~ 1 로 값의 범위를 지정

import tensorflow as tf
import matplotlib.pyplot as plt
import random
from tensorflow.examples.tutorials.mnist import input_data

learning_rate = 0.001
training_epoch = 15
display_step = 1
batch_size = 100

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

X = tf.placeholder('float', [None, 784])
Y = tf.placeholder('float', [None, 10])

W1 = tf.Variable(tf.random_normal([784, 256]))
W2 = tf.Variable(tf.random_normal([256, 256]))
W3 = tf.Variable(tf.random_normal([256, 128]))
W4 = tf.Variable(tf.random_normal([128, 10]))

b1 = tf.Variable(tf.random_normal([256]))
b2 = tf.Variable(tf.random_normal([256]))
b3 = tf.Variable(tf.random_normal([128]))
b4 = tf.Variable(tf.random_normal([10]))

L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))
L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), b2))
L3 = tf.nn.relu(tf.add(tf.matmul(L2, W3), b3))

hypothesis = tf.sigmoid(tf.matmul(L3, W4) + b4)

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

    #test model
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))

    #calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    print('Accuracy:', accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))

    #predict & show
    r = random.randint(0, mnist.test.num_examples - 1)
    print('Label:', sess.run(tf.argmax(hypothesis, 1), {X: mnist.test.images[r:r+1]}))
    print('Prediction:', sess.run(tf.argmax(hypothesis, 1), {X: mnist.test.images[r:r+1]}))

    '''#show the img
    plt.imshow(mnist.test.images[r:r+1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()'''