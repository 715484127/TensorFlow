import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#导入测试数据
mnist = input_data.read_data_sets("data/",one_hot = True)

#参数设置
numClasses = 10
inputSize = 784
numHiddenUnits = 50
trainingIterations = 10000
batchSize = 100

X = tf.placeholder(tf.float32, shape=[None,inputSize])
y = tf.placeholder(tf.float32, shape=[None,numClasses])

#参数初始化
W1 = tf.Variable(tf.truncated_normal([inputSize,numHiddenUnits],stddev=0.1))
B1 = tf.Variable(tf.constant(0.1),[numHiddenUnits])
W2 = tf.Variable(tf.truncated_normal([numHiddenUnits,numClasses],stddev=0.1))
B2 = tf.Variable(tf.constant(0.1),[numClasses])

#网络结构
hiddenLayerOutput = tf.matmul(X,W1) + B1
hiddenLayerOutput = tf.nn.relu(hiddenLayerOutput)
finalOutput = tf.matmul(hiddenLayerOutput,W2) + B2
finalOutput = tf.nn.relu(finalOutput)

