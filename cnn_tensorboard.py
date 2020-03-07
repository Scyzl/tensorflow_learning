'''
@Author: Scy
@Date: 2020-03-07 15:49:45
@LastEditTime: 2020-03-07 19:56:39
@LastEditors: Scy
@Description: 利用tensorboard查看所搭建的CNN结构
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
# 每个批次的大小
batch_size = 100
n_batch = mnist.train.num_examples // batch_size


# 参数概要
def variable_summaries(var):
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
    with tf.name_scope("stddev"):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


# 初始化权值
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)      # 生成一个截断的正态分布
    return tf.Variable(initial, name=name)


# 初始化偏置
def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


# 卷积层
def conv2d(x, W):
    """
    :param x: input tensor of shape '[batch, in_height, in_width, in_channels]'
    :param W: filter / kennel tensor of shape [filter_height, filter_width, in_channels, out_channels]
    stride[0] = stride[3] = 1, 其中，stride[1]代表x方向的步长，stride[2]代表y方向的步长
    padding： A string from: SAME, VALID
    :return:
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 池化层
def max_pool_2x2(x):
    # ksize [1, x, y, 1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')
    # 改变x的格式，转换为4D的向量 [batch, in_height, in_width, in_channels]
    with tf.name_scope('x_image'):
        x_image = tf.reshape(x, [-1, 28, 28, 1], name='x_image')

with tf.name_scope("Conv1"):
    # 初始化第一个卷积层的权重和偏置
    with tf.name_scope('W_conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32], name='W_conv1')    # 使用5*5的采样窗口，32个卷积核从一个平面抽取特征
    with tf.name_scope("b_conv1"):
        b_conv1 = bias_variable([32], name='b_conv1')       # 每一个卷积核一个偏置值

    # 把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
    with tf.name_scope('conv2d_1'):
        conv2d_1 = conv2d(x_image, W_conv1) + b_conv1
    with tf.name_scope('relu'):
        h_conv1 = tf.nn.relu(conv2d_1)
    with tf.name_scope('h_pool1'):
        h_pool1 = max_pool_2x2(h_conv1)         # 进行max-pooling

with tf.name_scope("Conv2"):
    # 初始化第二个卷积层的权重和偏置
    with tf.name_scope('W_conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64], name='W_conv2')       # 5*5的采样窗口，64个卷积核从32个平面抽取特征
    with tf.name_scope('b_conv2'):
        b_conv2 = bias_variable([64], name='b_conv2')
    
    # 把h_pool1和权值向量进行卷积，再加上偏置值，然后用relu激活
    with tf.name_scope('conv2d_2'):
        conv2d_2 = conv2d(h_pool1, W_conv2)
    with tf.name_scope('relu_2'):
        h_conv2 = tf.nn.relu(conv2d_2)
    with tf.name_scope('h_pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

"""
28*28的图片第一次卷积后还是28*28，第一次池化后变为14*14，
第二次卷积后为14*14，第二次池化后变为7*7，
故进行上边的操作后得到64张7*7的平面
"""

# 初始化第一个全连接层
with tf.name_scope('fc1'):
    with tf.name_scope('W_fc1'):
        W_fc1 = weight_variable([7*7*64, 1024], name='W_fc1')
    with tf.name_scope('b_fc1'):
        b_fc1 = bias_variable([1024], name='b_fc1')
    
    # 把池化层2的输出扁平化为1维
    with tf.name_scope('h_pool2_flat'):
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    # 求第一个全连接层的输出
    with tf.name_scope('wx_plus_b_fc1'):
        wx_plus_b_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
    with tf.name_scope('h_fc1'):
        h_fc1 = tf.nn.relu(wx_plus_b_fc1)

    with tf.name_scope('keep_prob'):
        keep_prob = tf.placeholder(tf.float32)
    with tf.name_scope('h_fc1_drop'):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# # 合并所有的summary
# merged = tf.summary.merge_all()

# 初始化第二个全连接层
with tf.name_scope('fc2'):
    with tf.name_scope('W_fc2'):
        W_fc2 = weight_variable([1024, 10], name='W_fc2')
    with tf.name_scope('b_fc2'):
        b_fc2 = bias_variable([10], name='b_fc2')   
    
    with tf.name_scope('wx_plus_b_fc2'):
        wx_plus_b_fc2 = tf.matmul(h_fc1_drop, W_fc2)
    with tf.name_scope('prediction'):
        # 计算输出
        prediction = tf.nn.softmax(wx_plus_b_fc2)

# 交叉熵代价函数
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
# 选择优化器
with tf.name_scope('train'):
    train_step = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('correct_prediction'):
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

with tf.compat.v1.Session() as sess:
    """
    sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()         # 合并所有的summary
    如果merged放在了变量初始化init前面，会导致merged为NoneType，故上面才是正确的顺序！！！
    """
    sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()         # 合并所有的summary
    train_writer = tf.summary.FileWriter('logs/train', sess.graph)
    test_writer = tf.summary.FileWriter('logs/test', sess.graph)
    for epoch in range(2001):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
        # print(summary)
        # print(type(summary))
        # 记录训练集计算的参数
        summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        train_writer.add_summary(summary, epoch)
        # 记录测试集计算的参数
        batch_xs, batch_ys = mnist.test.next_batch(batch_size)
        summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        test_writer.add_summary(summary, epoch)

        if epoch % 100 == 0:
            test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
            train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images[:10000], y: mnist.train.labels[:10000], keep_prob:1.0})
            print("Iter " + str(epoch) + ':\t' + "Test Accuarcy:    " + str(test_acc) + ";\tTrain Accuracy:  " + str(train_acc))
