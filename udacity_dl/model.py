import tensorflow as tf

class BasicCNNModel(object):
    def __init__(self, inupt_data, y_labels,
                 batch_size, image_size,
                 num_channels, num_labels,
                 patch_size, depth, num_hidden, stride,
                 tf_valid_dataset, tf_test_dataset):

            # Input data.
            self.input_data = inupt_data
            self.y_labels = y_labels

            # Variables.

            self.layer1_weights = tf.get_variable("layer1_W", shape=[patch_size, patch_size, num_channels, depth],
                    initializer=tf.contrib.layers.xavier_initializer())
            self.layer1_biases = tf.Variable(tf.zeros([depth]))
            self.layer2_weights = tf.get_variable("layer2_W", shape=[patch_size, patch_size, depth, depth],
                    initializer=tf.contrib.layers.xavier_initializer())
            self.layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))

            self.layer3_weights = tf.get_variable("layer3_W", shape=[image_size // 4 * image_size // 4 * depth, num_hidden],
                    initializer=tf.contrib.layers.xavier_initializer())
            self.layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
            self.layer4_weights = tf.get_variable("layer4_W", shape=[num_hidden, num_labels],
                    initializer=tf.contrib.layers.xavier_initializer())
            self.layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

            def batch_norm_wrapper(inputs, is_training, decay=0.999):

                # Small epsilon value for the BN transform
                epsilon = 1e-3

                scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
                beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
                pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
                pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

                if is_training:
                    batch_mean, batch_var = tf.nn.moments(inputs, [0])
                    train_mean = tf.assign(pop_mean,
                                           pop_mean * decay + batch_mean * (1 - decay))
                    train_var = tf.assign(pop_var,
                                          pop_var * decay + batch_var * (1 - decay))
                    with tf.control_dependencies([train_mean, train_var]):
                        return tf.nn.batch_normalization(inputs,
                                                         batch_mean, batch_var, beta, scale, epsilon)
                else:
                    return tf.nn.batch_normalization(inputs,
                                                     pop_mean, pop_var, beta, scale, epsilon)

            # Model
            def model(data, is_training):
                self.conv = tf.nn.conv2d(data, self.layer1_weights, [1, stride, stride, 1], padding='SAME')
                self.hidden = tf.nn.relu(self.conv + self.layer1_biases)

                # self.conv = tf.nn.conv2d(self.hidden, self.layer2_weights, [1, stride, stride, 1], padding='SAME')
                self.conv = tf.nn.max_pool(self.hidden, [1, stride, stride, 1], [1, stride, stride, 1], padding='VALID')

                self.hidden = tf.nn.relu(self.conv + self.layer2_biases)
                self.shape = self.hidden.get_shape().as_list()
                self.reshape = tf.reshape(self.hidden, [self.shape[0], self.shape[1] * self.shape[2] * self.shape[3]])

                z1 = tf.matmul(self.reshape, self.layer3_weights) + self.layer3_biases
                bz1 = batch_norm_wrapper(z1,is_training)
                self.hidden = tf.nn.relu(bz1)
                logits = tf.matmul(self.hidden, self.layer4_weights) + self.layer4_biases
                return logits

            self.logits = model(self.input_data, is_training=True)
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.y_labels, logits=self.logits))

            self.train_prediction = tf.nn.softmax(self.logits)
            self.valid_prediction = tf.nn.softmax(model(tf_valid_dataset, is_training=True))
            self.test_prediction = tf.nn.softmax(model(tf_test_dataset, is_training=True))

