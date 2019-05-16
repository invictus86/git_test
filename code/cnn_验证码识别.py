# coding=utf-8
import tensorflow as tf
import os
import pandas as pd
from captha_model import model

#1. 准备数据，准备特征值
def read_jpg():
    #1.1 读取图片
    #1.1.1 构建文件名队列
    path = "../data/GenPics"
    file_list = [os.path.join(path,i) for i in os.listdir(path) if i.endswith("jpg")]
    file_queue = tf.train.string_input_producer(file_list)
    #1.1.2 读取图片
    reader = tf.WholeFileReader()
    key,value = reader.read(file_queue)
    #1.1.3 解码
    image = tf.image.decode_jpeg(value,channels=3)
    print(image)
    image_reshaped = tf.reshape(image,(20,80,3))
    # print(image_reshaped)
    #1.1.4 批处理
    image_batch,filename_batch = tf.train.batch([image_reshaped,key],batch_size=50,num_threads=2,capacity=500)
    return image_batch,filename_batch
#2. 准备目标值
def get_csv_data():
    path = "../data/GenPics/labels.csv"
    df = pd.read_csv(path,names=["index","chars"],index_col="index")
    # 在df中添加一列labels [10,20,22,5]
    df["labels"] = None
    for i,_data in df.iterrows():
        temp_list = []
        for char in _data["chars"]:
            temp_list.append(ord(char)-ord("A"))
        df.loc[i,"labels"] = temp_list
    # print(df)
    return df

def filenames_2_labels(filenames,csv_data):
    #filenames :[1023,24,32...]
    filenames = [i.decode().split("/")[-1].split(".")[0] for i in filenames]
    # print(filenames)
    # print(type(filenames[0]))
    lable_list = [csv_data.loc[int(i),"labels"] for i in filenames]
    return lable_list

def main(argv):
    #1. 读取图片和文件名
    image_batch, filename_batch = read_jpg()
    #2. 读取csv文件
    csv_data = get_csv_data()
    #4. 定义模型
    x_data, y_true, y_predict, loss, train_op, accuracy = model()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess,coord)

        #恢复
        checkpoint = tf.train.latest_checkpoint("./")
        if checkpoint:
            saver.restore(sess,checkpoint)
        for i in range(2000):
            # image :[50,20,80,3] filenames[50]
            images,filenames = sess.run([image_batch, filename_batch])
            #3. 根据filename获取labels
            labels = filenames_2_labels(filenames,csv_data)
            labels_onehot = tf.one_hot(labels,axis=-1,depth=26)  #[None,4,26]
            labels_reshaped = tf.reshape(labels_onehot,[-1,4*26]).eval()
            _,_loss,_acc  = sess.run([train_op,loss,accuracy],feed_dict={x_data:images,y_true:labels_reshaped})
            print(i,"--损失：",_loss,"--准确率：",_acc)

            # 实现会话的保存
            if (i+1)%100==0:
                saver.save(sess,"./captcha.ckpt")

        coord.request_stop()
        coord.join(threads)
if __name__ == '__main__':
    tf.app.run()
    # get_csv_data()