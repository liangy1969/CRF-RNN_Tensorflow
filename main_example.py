import tensorflow as tf
import numpy as np
import scipy.misc

import CRFRNN


if __name__ == '__main__':

    # first read the image
    img = scipy.misc.imread('tabby_cat.png')

    # the size of image
    H, W, n_channel = img.shape

    # add dimension to img
    img = np.expand_dims(img, axis=0)

    # the number of classes
    n_class = 3

    # the number of batches
    n_batch = 1

    # build up place holders
    # the place holder for pixel labels
    label_holder = tf.placeholder(dtype=tf.float32, shape=[n_batch, H, W, n_class])
    # the place holder for unary potential of each pixel
    # should replace this place holder with the last layer of FCN in practice
    potential_holder = tf.placeholder(dtype=tf.float32, shape=[n_batch, H, W, n_class])
    # reshape the potential holder
    potential_tensor = tf.reshape(potential_holder, shape=[-1, H * W, n_class])

    # build up the components of CRF_RNN
    # the parameters needed for the filters
    gamma = 30.0  # spatial
    sigma_bilateral = np.array([1.0] * n_channel + [1.0] * 2)  # bilateral

    # first the transformation vector for spatial and bilateral filters
    # the spatial filter is global
    spatial_dict = CRFRNN.spatial_global(H, W, gamma)
    # the tensor of spatial filter vectors
    spatial_tensor_dict = {}
    spatial_tensor_dict['splat'] = tf.constant(spatial_dict['splat'], dtype=tf.int64)
    spatial_tensor_dict['weight'] = tf.constant(spatial_dict['weight'], dtype=tf.float32)
    spatial_tensor_dict['n_lattice'] = tf.constant(spatial_dict['n_lattice'], dtype=tf.int64, shape=[])
    spatial_tensor_dict['blur'] = tf.constant(spatial_dict['blur'], dtype=tf.int64)
    # the tensor of bilateral filter vectors
    bilateral_tensor_dict = {}
    bilateral_tensor_dict['splat'] = []
    bilateral_tensor_dict['weight'] = []
    bilateral_tensor_dict['n_lattice'] = []
    bilateral_tensor_dict['blur'] = []
    # create place holders for each image in the batch
    for i in range(n_batch):
        bilateral_tensor_dict['splat'].append(
            tf.placeholder(dtype=tf.int64, shape=[(n_channel + 2 + 1) * H * W]))
        bilateral_tensor_dict['weight'].append(
            tf.placeholder(dtype=tf.float32, shape=[(n_channel + 2 + 1) * H * W]))
        bilateral_tensor_dict['n_lattice'].append(tf.placeholder(dtype=tf.int64, shape=[]))
        bilateral_tensor_dict['blur'].append(tf.placeholder(dtype=tf.int64, shape=[None]))

    # define the RNN-CRF
    rnn_crf = CRFRNN.CRFRNN(n_channel=n_channel, n_class=n_class, n_batch=n_batch,
                            H=H, W=W, name='CRFRNN', n_unit=10)

    # build up RNN-CRF
    logits_output = rnn_crf.build(potential_tensor, spatial_tensor_dict, bilateral_tensor_dict)

    # reshape output
    logits_output_reshape = tf.reshape(logits_output, shape=[n_batch, H, W, n_class])

    # start a session
    with tf.Session() as sess:

        # randomly initialize the CRF-RNN
        sess.run(tf.global_variables_initializer())

        # run the CRF-RNN
        # generate random labels and potentials for demonstration
        pixel_class = np.random.randint(n_class, size=(1, H, W),dtype='uint8')
        label = (np.arange(n_class, dtype='uint8') == pixel_class[:, :, :, None])
        potential = np.random.rand(1, H, W, n_class)

        # compute the transformation matrix of filters manually from the original image
        # calculate the bilateral filter
        bilateral_dict = CRFRNN.bilateral_batch(img, sigma_bilateral)

        # construct the feed dict
        feed_dict = {potential_holder: potential, label_holder: label}

        for j in range(n_batch):
            feed_dict[bilateral_tensor_dict['splat'][j]] = bilateral_dict['splat'][j]
            feed_dict[bilateral_tensor_dict['weight'][j]] = bilateral_dict['weight'][j]
            feed_dict[bilateral_tensor_dict['n_lattice'][j]] = bilateral_dict['n_lattice'][j]
            feed_dict[bilateral_tensor_dict['blur'][j]] = bilateral_dict['blur'][j]

        logits_val = sess.run([logits_output_reshape], feed_dict=feed_dict)


