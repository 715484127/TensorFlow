import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 生成1000个随机分布的点
num_points = 1000
vectors_set = []
for i in range(num_points):
    x1 = np.random.normal(0.0,0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0,0.03)
    vectors_set.append([x1,y1])
x_data = [v[0] for v in vectors_set]
y_dara = [v[1] for v in vectors_set]
plt.scatter(x_data,y_dara,c='r')
plt.show()

# 生成一维的W矩阵 取值是[-1,1]之间的随机数
W = tf.Variable(tf.random_uniform([1],-1.0,1.0),name='W')
# 生成一维的b矩阵 初始值是0
b = tf.Variable(tf.zeros([1]))
# 经过计算得出预估值y
y = W * x_data + b

# 以预估值y和实际值y_dara之间的均方差作为损失
loss = tf.reduce_mean(tf.square(y - y_dara))
# 采用梯度下降法优化参数
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 训练过程就是最小化损失值
train = optimizer.minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print("W = ",sess.run(W), "b = ",sess.run(b) ,"loss = ",sess.run(loss))
# 执行20次训练
for step in range(20):
    sess.run(train)
    print("W = ", sess.run(W), "b = ", sess.run(b), "loss = ", sess.run(loss))

plt.scatter(x_data,y_dara,c='r')
plt.plot(x_data,sess.run(W)*x_data+sess.run(b))
plt.show()
