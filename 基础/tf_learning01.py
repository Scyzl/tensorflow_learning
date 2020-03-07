'''
@Author: Scy
@Date: 2020-03-01 17:06:48
@LastEditTime: 2020-03-07 19:53:38
@LastEditors: Scy
@Description: the first course of learning tensorflow
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# m1 = tf.constant([[2, 3]])
# m2 = tf.constant([[2], [3]])
# product = tf.matmul(m1, m2)
# print(product)
#
# with tf.Session() as sess:
#     result = sess.run(product)
#     print(result)

# 使用numpy生成200个随机点
x_data = np.linspace(-0.5, 0.5, 500)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise
# print(x_data.dtype, y_data.dtype, noise.dtype)

# 定义两个placeholder
x = tf.compat.v1.placeholder(tf.float32, [None, 1])
y = tf.compat.v1.placeholder(tf.float32, [None, 1])

# 定义神经网络中间层
weights_l1 = tf.Variable(tf.random.normal([1, 20], dtype=tf.float32))
biases_l1 = tf.Variable(tf.zeros([1, 20], dtype=tf.float32))
# print(weights_l1.dtype, biases_l1.dtype)
wx_plus_b_l1 = tf.matmul(x_data, weights_l1) + biases_l1
l1 = tf.nn.tanh(wx_plus_b_l1)

# 定义神经网络输出层
weights_l2 = tf.Variable(tf.random.normal([20, 1], dtype=tf.float32))
biases_l2 = tf.zeros([1, 1], dtype=tf.float32)
wx_plus_b_l2 = tf.matmul(l1, weights_l2) + biases_l2
prediction = tf.nn.tanh(wx_plus_b_l2)

# 二次代价函数
loss = tf.reduce_mean(tf.square(y - prediction))
# 梯度下降法训练优化器
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})      # 喂入数据集进行训练

    # 获得预测值
    prediction_value = sess.run(prediction, feed_dict={x: x_data})
    # 画图
    plt.figure()
    plt.scatter(x_data, y_data)         # 散点图
    plt.plot(x_data, prediction_value, 'r', lw=5)
    plt.show()
