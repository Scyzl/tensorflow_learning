from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def train():
    # 载入数据
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

    # 每个批次的大小
    batch_size = 100
    # 计算一共有多少个批次
    n_batch = mnist.train.num_examples

    # 命名空间
    with tf.name_scope('input'):
        # 定义两个placeholder
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y = tf.placeholder(tf.float32, [None, 10], name='y-input')

    with tf.name_scope('layer'):
        # 创建一个简单的神经网络
        with tf.name_scope('weights'):
            W = tf.Variable(tf.zeros([784, 10], dtype=tf.float32))
        with tf.name_scope('bais'):
            b = tf.Variable(tf.zeros([10], dtype=tf.float32))
        with tf.name_scope('wx_plus_b'):
            wx_plus_b = tf.matmul(x, W) + b
        with tf.name_scope('softmax'):
            prediction = tf.nn.softmax(wx_plus_b)

    # 定义二次代价函数
    with tf.name_scope('loss'):
        # loss = tf.reduce_mean(tf.square(y - prediction))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

    # 初始化变量
    init = tf.global_variables_initializer()

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            # 结果存放在一个bool类型的列表中
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))   # argmax 返回一维张量中最大值所在位置
        with tf.name_scope('accuracy'):
            # 求准确率
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))      # tf.cast(a, dtype0) 将a的dtype转换成dtype0

    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter('logs/', sess.graph)
        for epoch in range(1):     # 迭代周期
            for batch in range(n_batch):     # 每个周期内的训练批次
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

            acc = sess.run(accuracy, feed_dict=({x: mnist.test.images, y: mnist.test.labels}))
            print("Iter" + str(epoch) + "Test accuracy: " + str(acc))


if __name__ == '__main__':
    train()
