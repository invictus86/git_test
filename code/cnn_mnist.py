# coding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from cnn_model import model


#1. 准备model
x_data, y_true, y_predict, loss, train_op, accuracy = model()

#1.1 准备数据
mnist = input_data.read_data_sets("./data",one_hot=True)

#2. 开启会话执行
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    image_batch,label_batch = mnist.train.next_batch(batch_size=50)
    for i in range(2000):
        _, _loss, _accuracy = sess.run([train_op,loss,accuracy],feed_dict={x_data:image_batch,y_true:label_batch})
        print("i:",i," loss:",_loss," acc:",_accuracy)
