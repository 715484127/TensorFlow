import tensorflow as tf

w = tf.Variable([[0.5,1.0]])
x = tf.Variable([[2.0],[1.0]])
y = tf.matmul(w,x)
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print(y.eval())

norm = tf.random_normal([2,3],mean=-1,stddev=4)
c = tf.constant([[1,2],[3,4],[5,6]])
shuff = tf.random_shuffle(c)
with tf.Session() as sess:
    sess.run(norm)
    print(norm.eval())
    sess.run(shuff)
    print(shuff.eval())

state = tf.Variable(0)
new_state = tf.add(state,tf.constant(1))
update = tf.assign(state,new_state)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(state.eval())
    for _ in range(3):
        sess.run(update)
        print(state.eval())

a = tf.constant(5)
b = tf.constant(10)
a1 = tf.add(a,b,name="add")
a2 = tf.div(b,a,name="div")
with tf.Session() as sess:
    print(sess.run(a1))
    print(sess.run(a2))

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1,input2)
with tf.Session() as sess:
    print(sess.run([output],feed_dict={input1:[7.],input2:[2.]}))
