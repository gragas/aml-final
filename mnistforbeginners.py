import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def main():
    # Load the input data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Setup variables and placeholders
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # Implement our model
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    
    # Placeholder to input the correct answers
    y_ = tf.placeholder(tf.float32, [None, 10])
    
    # Implement cross-entropy
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    # Apply an optimization algorithm to reduce the cost
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    
    # Initialize variables
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    # Now let's train!
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Evaluate our model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels}))

if __name__ == "__main__":
    main()
