# coding: utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector


# 载入数据集
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

max_step = 1001     # 运行次数
image_num = 3000    # 图片数量
DIR = 'F:/Study/CS/TensorFlow/'     # 文件路径

# 定义会话
sess = tf.Session()

# 加载图片
embedding = tf.Variable(tf.stack(mnist.test.images[:image_num]), trainable=False, name='embedding')


# 参数概要
def variable_summary(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)      # 直方图


with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')

# 显示图片
with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

with tf.name_scope('layer'):
    with tf.name_scope('weights'):
        W = tf.Variable(tf.zeros([784, 10], dtype=tf.float32), name='W')
        variable_summary(W)
    with tf.name_scope('bias'):
        b = tf.Variable(tf.zeros([10], dtype=tf.float32), name='bias')
        variable_summary(b)
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x, W) + b
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(wx_plus_b)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(0.5).minimize(loss)

# 初始化变量
sess.run(tf.global_variables_initializer())

with tf.name_scope('accuracy'):
    with tf.name_scope('accuracy_prediction'):
        accuracy_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))   # argmax返回一维张量中最大的值所在的位置
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(accuracy_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

# 产生metadata文件
if tf.gfile.Exists(DIR + "projector/projector/metadata.tsv"):
    tf.gfile.DeleteRecursively(DIR + "projector/projector/metadata.tsv")
with open(DIR + "projector/projector/metadata.tsv", 'w') as f:
    labels = sess.run(tf.argmax(mnist.test.labels[:], 1))
    for i in range(image_num):
        f.write(str(labels[i]) + '\n')

# 合并所有的summary
merged = tf.summary.merge_all()

projector_writer = tf.summary.FileWriter(DIR + 'projector/projector/', sess.graph)
saver = tf.train.Saver()
config = projector.ProjectorConfig()
embed = config.embeddings.add()
embed.tensor_name = embedding.name
embed.metadata_path = DIR + 'projector/projector/metadata.tsv'
embed.sprite.image_path = DIR + 'projector/data/mnist_10k_sprite.png'
embed.sprite.single_image_dim.extend([28, 28])
projector.visualize_embeddings(projector_writer, config)

for i in range(max_step):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    summary, _ = sess.run([merged, train_step],
                          feed_dict={x: batch_xs, y: batch_ys},
                          options=run_options,
                          run_metadata=run_metadata)
    projector_writer.add_run_metadata(run_metadata, 'step%03d' % i)
    projector_writer.add_summary(summary, i)

    if i % 100 == 0:
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter  " + str(i) + "\t" + "Testing Accuracy:  " + str(acc))

saver.save(sess, DIR + 'projector/projector/a_model.ckpt', global_step=max_step)
projector_writer.close()
sess.close()
