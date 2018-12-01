#!/usr/bin/python
# --*-- coding: utf-8 --*--

import tensorflow as tf
import math
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import random

data_index = 0


def data_precess(data):
    data = np.array(data)
    data_stand = preprocessing.StandardScaler().fit(data)
    data = data_stand.transform(data)
    return data, data_stand


def gen_batch(batch_size, data, labels, data_shuffle):
    global data_index
    inputs = []
    labels_to_vec = []
    for i in range(batch_size):
        index = (data_index+i) % (len(data)-300)
        inputs.append(data[data_shuffle[index]])
        if labels[data_shuffle[index]] > 0:
            temp = [1, 0]
        else:
            temp = [0, 1]
        labels_to_vec.append(temp)
    data_index += 1
    return inputs, labels_to_vec


def gen_batch_test(data, act_val, data_shuffle):
    inputs = []
    labels = []
    for i in range(len(data)-300, len(data)):
        inputs.append(data[data_shuffle[i]])
        if act_val[data_shuffle[i]] > 0:
            temp = [1, 0]
        else:
            temp = [0, 1]
        labels.append(temp)
    return inputs, labels


def build_graph(batch_size, data, act_val, data_shuffle):
    graph = tf.Graph()
    with graph.as_default():
        # 数据预处理
        data_process, data_stand,  = data_precess(data)
        len_data = len(data_process[0])
        print len_data
        # print len_data
        # 输入层
        train_inputs = tf.placeholder(tf.float32, shape=[None, len_data])
        train_labels = tf.placeholder(tf.float32, shape=[None, 2])
        hidden_units = [100, 50, 20]
        # 第一层
        weights_1 = tf.Variable(
            tf.truncated_normal([len_data, hidden_units[0]], stddev=1.0 / math.sqrt(len(data))
                                )
        )
        bias_1 = tf.Variable(tf.zeros([hidden_units[0]]))
        y_1 = tf.matmul(train_inputs, weights_1) + bias_1
        y_1_act = tf.nn.elu(y_1)
        '''
        # 第二层
        weights_2 = tf.Variable(
            tf.truncated_normal([hidden_units[0], hidden_units[1]], stddev=1.0 / math.sqrt(len(data))
                                )
        )
        bias_2 = tf.Variable(tf.zeros([hidden_units[1]]))
        y_2 = tf.matmul(y_1_act, weights_2) + bias_2
        y_2_act = tf.nn.relu(y_2)

        # 第三层
        weights_3 = tf.Variable(tf.truncated_normal([hidden_units[1], hidden_units[2]], stddev=1.0 / math.sqrt(len(data))
                                                    )
                                )
        bias_3 = tf.Variable(tf.zeros([hidden_units[2]]))
        y_3 = tf.matmul(y_2_act, weights_3) + bias_3
        y_3_act = tf.nn.relu(y_3)
        '''
        # 第4层
        weights_4 = tf.Variable(tf.truncated_normal([hidden_units[0], 2], stddev=1.0 / math.sqrt(len(data))
                                                    )
                                )
        bias_4 = tf.Variable(tf.zeros([2]))
        y_4 = tf.matmul(y_1_act, weights_4) + bias_4
        y_process = tf.nn.softmax(y_4)
        # 正则化
        reguralization = tf.norm(weights_1, 2) + tf.norm(weights_4, 2)  # + tf.norm(weights_3, 2) + tf.norm(weights_4, 2)

        # 学习率
        l_r = 0.01
        learning_rate = tf.placeholder(tf.float64)

        # 损失函数
        loss = -tf.reduce_mean(train_labels*tf.log(y_process)) + 5e-4*reguralization
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        # 初始化变量
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            x = []
            y1 = []
            y2 = []
            for step in range(150000):
                # learn_rate = l_r*math.pow(0.99, step/100)
                learn_rate = l_r

                inputs, labels = gen_batch(batch_size, data_process, act_val, data_shuffle)
                # inputs_d = combine_data(inputs)
                _, l = sess.run([train_step, loss], feed_dict={train_inputs: inputs,
                                                               train_labels: labels,
                                                               learning_rate: learn_rate})

                if step % 1000 == 0:
                    inputs_test, labels_test = gen_batch_test(data_process, act_val, data_shuffle)
                    l_test, reg = sess.run([loss, reguralization], feed_dict={train_inputs: inputs_test,
                                                       train_labels: labels_test})
                    print "运行步数： ",step, " 损失值: ", l, "  测试损失", l_test,"  权重 " , 5e-4*reg, " 学习率 ", learn_rate
                    x.append(step)
                    y1.append(l)
                    y2.append(l_test)
            plt.plot(np.array(x), np.array(y1))
            plt.plot(np.array(x), np.array(y2))
            plt.show()

            # 预测
            # 预测文件
            file_name_predict = "../data/data_pre/test_vec_v1.csv"
            fr = open(file_name_predict, 'r')
            print(fr.readline())
            data_test, instanceID = read_test_file(fr)

            result = dict()
            while len(data_test) > 0:
                test_input = data_stand.transform(data_test)
                y_test = sess.run(y_process, feed_dict={train_inputs: test_input})
                for i in range(len(instanceID)):
                    result[int(instanceID[i])] = y_test[i][0]
                data_test, instanceID = read_test_file(fr)

            # 排序
            sort_key = result.keys()
            sort_key.sort()

            # 输出文件
            file_name_output = "../result/2017_06_04_submission_evening.csv"
            fr_output = open(file_name_output, "w")
            fr_output.writelines("instanceID,predict" + "\n")
            # 输出
            for key in sort_key:
                pre_data = result[key]
                line = ",".join([str(key), str(pre_data)]) + "\n"
                fr_output.writelines(line)

            fr.close()
            fr_output.close()


def read_file(file_name):
    fr = open(file_name, 'r')
    print(fr.readline())
    data = []
    act_val = []
    for line in fr.readlines():
        line = line.replace('"', '').replace("\n", "").split(",")
        data_line = []
        act_val.append(float(line[1]))
        for i in range(2, len(line)):
            data_line.append(float(line[i]))
        data.append(data_line)
    return data, act_val


def read_test_file(file_name):
    fr = open(file_name, "r")
    print(fr.readline())
    data = []
    test_id = []
    for line in fr.readlines():
        line = line.replace('"', '').replace("\n", "").split(",")
        data_line = []
        test_id.append([line[0]])
        for i in range(1, len(line)):
            data_line.append(float(line[i]))
        data.append(data_line)
    return data, test_id


def main():
    data, act_val = read_file("../data/data_pre/train_vec_v1.csv")
    batch_size = 64
    build_graph(batch_size, data, act_val)


if __name__ == "__main__":
    main()


