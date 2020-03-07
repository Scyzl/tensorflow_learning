'''
@Author: Scy
@Date: 2020-03-03 16:42:28
@LastEditTime: 2020-03-07 19:53:01
@LastEditors: Scy
@Description: 
'''
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def train():
    # 载入数据
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

    # 每个批次的大小
    batch_size = 100
    # 计算一共有多少个批次
    n_batch = mnist.train.num_examples

    # 定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)

    # 创建神经网络输入层
    W1 = tf.Variable(tf.truncated_normal([784, 2000], stddev=0.1, dtype=tf.float32))
    b1 = tf.Variable(tf.zeros([2000], dtype=tf.float32) + 0.1)
    L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
    L1_drop = tf.nn.dropout(L1, keep_prob)

    # 创建隐藏层
    W2 = tf.Variable(tf.truncated_normal([2000, 1000], stddev=0.1, dtype=tf.float32))
    b2 = tf.Variable(tf.zeros([1000], dtype=tf.float32) + 0.1)
    L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
    L2_drop = tf.nn.dropout(L2, keep_prob)

    W3 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.1, dtype=tf.float32))
    b3 = tf.Variable(tf.zeros([10], dtype=tf.float32) + 0.1)
    # L3 = tf.nn.tanh(tf.matmul(L2_drop, W3) + b3)
    # L3_drop = tf.dropout(L3, keep_prob)

    prediction = (tf.matmul(L2_drop, W3) + b3)

    # 定义二次代价函数
    # loss = tf.reduce_mean(tf.square(y - prediction))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

    # 初始化变量
    init = tf.global_variables_initializer()

    # 结果存放在一个bool类型的列表中
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))   # argmax 返回一维张量中最大值所在位置
    # 求准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))      # tf.cast(a, dtype0) 将a的dtype转换成dtype0

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(21):     # 迭代周期
            for batch in range(n_batch):     # 每个周期内的训练批次
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})

            test_acc = sess.run(accuracy, feed_dict=({x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))
            train_acc = sess.run(accuracy, feed_dict=({x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0}))
            print("Iter" + str(epoch) + '\t\t' + "Test accuracy: " + str(test_acc) + '\t\t'
                  + "Train accuracy: " + str(train_acc))


if __name__ == '__main__':
    train()
