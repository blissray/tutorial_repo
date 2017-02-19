import tensorflow as tf
from six.moves import cPickle as pickle
import numpy as np

from udacity_dl.model import BasicCNNModel

def main():
    params = dict()
    params["batch_size"] = 128
    params["image_size"] = 28
    params["num_channels"] = 1
    params["num_labels"] = 10
    params["patch_size"] = 5
    params["depth"] = 16
    params["num_hidden"] = 64
    params["stride"] = 2
    params["num_steps"] = 1001
    train(params)

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

def train(params):
    pickle_file = 'notMNIST.pickle'

    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']

    import numpy as np

    def reformat(dataset, labels):
        dataset = dataset.reshape(
            (-1, params["image_size"], params["image_size"], params["num_channels"])).astype(np.float32)
        labels = (np.arange(params["num_labels"]) == labels[:, None]).astype(np.float32)
        return dataset, labels

    train_dataset, train_labels = reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)


    x_dataset = tf.placeholder(
        tf.float32, shape=[params["batch_size"], params["image_size"], params["image_size"], params["num_channels"]])
    y_labels = tf.placeholder(tf.float32, shape=[params["batch_size"], params["num_labels"]])

    model = BasicCNNModel(
        inupt_data=x_dataset, y_labels=y_labels,
        batch_size=params["batch_size"], image_size=params["image_size"] , num_channels=params["num_channels"],
        num_labels=params["num_labels"], patch_size=params["patch_size"], depth=params["depth"] , num_hidden=params["num_hidden"],
        stride=params["stride"], tf_test_dataset= test_dataset, tf_valid_dataset=valid_dataset
    )

    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(model.loss)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for step in range(params["num_steps"]):
            offset = (step * params["batch_size"]) % (train_labels.shape[0] - params["batch_size"])
            batch_data = train_dataset[offset:(offset + params["batch_size"]), :, :, :]
            batch_labels = train_labels[offset:(offset + params["batch_size"]), :]

            feed_dict = {x_dataset: batch_data, y_labels: batch_labels}
            _, l, train_prediction = sess.run(
                [optimizer, model.loss, model.train_prediction], feed_dict=feed_dict)

            if (step % 50 == 0):
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % accuracy(train_prediction, batch_labels))
                print('Validation accuracy: %.1f%%' % accuracy(
                    model.valid_prediction.eval(), valid_labels))
        print('Test accuracy: %.1f%%' % accuracy(model.test_prediction.eval(), test_labels))

if __name__ == '__main__':
    main()