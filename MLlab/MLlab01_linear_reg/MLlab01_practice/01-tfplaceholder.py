#tensorflow 의 placeholder를 이용하여 add, mul 함수를 만들어라.

import tensorflow as tf

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a,b)
mul = tf.mul(a,b)

with tf.Session() as sess:
    print('Addiction : %i' % sess.run(add, feed_dict={a: 2, b: 3}))
    print('Multiplication : %i' % sess.run(mul, feed_dict={a: 4, b:10}))