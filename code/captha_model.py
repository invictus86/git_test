# coding=utf-8
import tensorflow as tf


def model():
    """
    :return: 特征值，目标值，预测值，损失，准确率，train_op
    """
    # 1. 准备数据
    with tf.variable_scope("prepar_data"):
        #输入的数据 x_data [None,20,80,3]  y_true [None,4*26]
        x_data = tf.placeholder(dtype=tf.float32, shape=[None, 20,80,3], name="x_data")
        y_true = tf.placeholder(dtype=tf.float32, shape=[None, 4*26], name="y_true")

    # 2. 第一层卷积
    with tf.variable_scope("conv1"):
        # 2.1 卷积层
        # fileter [3,3,3,32]
        conv1_filter = tf.Variable(initial_value=tf.random_normal([3, 3, 3, 32],mean=0,stddev=0.01), name="conv1_filter")
        conv1_bias = tf.Variable(initial_value=tf.random_normal([32],mean=0,stddev=0.01), name="conv1_bias")
        # 输出 [None, 20, 80, 32]
        conv1_output = tf.nn.conv2d(input=x_data, filter=conv1_filter, strides=[1, 1, 1, 1],
                                    padding="SAME") + conv1_bias
        # 2.2 激活层   # 输出 [None, 20, 80, 32]
        conv1_relu = tf.nn.relu(conv1_output)
        # 2.3 池化层   # 输出 [None, 10, 40, 32]
        conv1_maxpool = tf.nn.max_pool(conv1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 3. 第二层卷积
    with tf.variable_scope("conv2"):
        # 3.1 卷积层
        # fileter [3,3,32,64]
        conv2_filter = tf.Variable(initial_value=tf.random_normal([3, 3, 32, 64],mean=0,stddev=0.01), name="conv2_filter")
        conv2_bias = tf.Variable(initial_value=tf.random_normal([64],mean=0,stddev=0.01), name="conv2_bias")
        # 输出 [None, 10, 40, 64]
        conv2_output = tf.nn.conv2d(input=conv1_maxpool, filter=conv2_filter, strides=[1, 1, 1, 1],
                                    padding="SAME") + conv2_bias
        # 3.2 激活层   # 输出 [None, 10, 40, 64]
        conv2_relu = tf.nn.relu(conv2_output)
        # 3.3 池化层   # 输出 [None, 5, 20, 64]
        conv2_maxpool = tf.nn.max_pool(conv2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 4. 全连接层
    with tf.variable_scope("fc"):
        fc_input = tf.reshape(conv2_maxpool, [-1, 5*20*64])  #TODO
        fc_w = tf.Variable(initial_value=tf.random_normal([5*20*64, 4*26],mean=0,stddev=0.01), name="fc_w")
        fc_bias = tf.Variable(initial_value=tf.random_normal([4*26],mean=0,stddev=0.01), name="fc_w")
        # y_predict [None,4*26]
        y_predict = tf.matmul(fc_input, fc_w) + fc_bias

    # 5. 计算得到损失，优化损失
    with tf.variable_scope("train"):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_predict))
        # train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
        train_op = tf.train.AdamOptimizer(0.0001).minimize(loss)

    # 6. 计算准确率
    with tf.variable_scope("get_accuracy"):
        # y_predict,y_true ---> [None,4,26]
        y_true_reshaped = tf.reshape(y_true,[-1,4,26])
        y_predict_reshaped = tf.reshape(y_predict,[-1,4,26])
        # [[True,False,False,True],[True,True,True,True]....]
        equal_list = tf.equal(tf.argmax(y_true_reshaped, axis=-1), tf.argmax(y_predict_reshaped, axis=-1))
        accuracy = tf.reduce_mean(tf.cast(tf.reduce_all(equal_list,axis=-1), tf.float32))

    return x_data, y_true, y_predict, loss, train_op, accuracy
