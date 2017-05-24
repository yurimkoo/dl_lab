#dropout
#overfitting을 방지하기 위해 사용
#무작위로 몇 개의 노드를 작동시키지 않으면서 여러번 학습하고 마지막에는 전체 노드 사용해서 output
#dropout_rate = tf.placeholder('float')
#_L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), B1))
#L1 = tf.nn.dropout(_L1, dropout_rate)

#ensemble
#다양한 학습모델을 만들어서 마지막에 합침. 성능 향상 2-4,5%

