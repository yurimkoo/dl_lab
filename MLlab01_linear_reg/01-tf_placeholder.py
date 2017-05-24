import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)
c = a+b

sess = tf.Session()

print(sess.run(c))

#Placeholder : 기본 함수 구성과 비슷함

x = tf.placeholder(tf.int16)
y = tf.placeholder(tf.int16)

add = tf.add(a,b)
mul = tf.mul(a,b)

with tf.Session() as sess:
    print("Addiction : %i " % sess.run(add, feed_dict={a: 2, b: 3}))
    print("Multiplication : %i " % sess.run(mul, feed_dict={a: 2, b: 5}))