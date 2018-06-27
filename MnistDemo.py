import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

print("导入包成功")
print("MNIST 数据下载中......")
mnist = input_data.read_data_sets('data/', one_hot=True)

print(" 类型是 %s " % (type(mnist)))
print(" 训练数据有 %d " % (mnist.train.num_examples))
print(" 测试数据有 %d " % (mnist.test.num_examples))

# 查看数据集内容
trainimg = mnist.train.images
trainlabel = mnist.train.labels
nsample = 5
randidx = np.random.randint(trainimg.shape[0], size=nsample)
for i in randidx:
    curr_img = np.reshape(trainimg[i,:],(28,28))
    curr_label = np.argmax(trainlabel[i,:])
    plt.matshow(curr_img, cmap=plt.get_cmap('gray'))
    plt.show()
