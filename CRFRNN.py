"""
 Tensorflow implementation of CRF-RNN
 See paper 'Fast High-Dimensional Filtering Using the Permutohedral Lattice' for algorithm interpretation
 See main_example.py for demonstration of usage
  Author: Yuan Liang @ Michigan State University
  email: liangy11@msu.edu
"""
import permutohedral
import numpy as np
import tensorflow as tf


class CRFRNN:

    def __init__(self, n_channel, n_class, n_batch, H, W, name, n_unit):
        """
        CRF-RNN
        :param n_channel: number of channels of image
        :param n_class: number of classes
        :param n_batch: batch size
        :param H: height of image
        :param W: weight of image
        :param name: the variable scope
        :param n_unit: the number of RNN units
        """

        self.d = n_channel
        self.n_class = n_class
        self.H = H
        self.W = W
        self.n_pixel = H*W
        self.name = name
        self.n_batch = n_batch
        self.n_unit = n_unit

    def build(self, U_tensor_batch, spatial_filter_dict, bilateral_filter_batch):

        with tf.variable_scope(self.name):

            # define the weights
            spatial_weight = tf.get_variable('spatial_weight', shape=[self.n_class], dtype=tf.float32,
                                                initializer = tf.constant_initializer(3))
            bilateral_weight = tf.get_variable('bilateral_weight', shape=[self.n_class], dtype=tf.float32,
                                                    initializer=tf.constant_initializer(5))
            compatible_weight = tf.get_variable('compatible_weight', shape=[self.n_class, self.n_class],
                                                     initializer=tf.constant_initializer(np.identity(self.n_class)))

            # unpack the batch tensor
            U_tensor_list = tf.unstack(U_tensor_batch, num = self.n_batch)
            # preprocess filters:
            splat_spatial, blur_spatial = self.transformation_tensor(spatial_filter_dict['splat'], spatial_filter_dict['weight'],
                                                                     spatial_filter_dict['blur'], spatial_filter_dict['n_lattice'], 2, n=self.n_pixel)
            # the spatial filtering is identical for each image patch

            splat_bilateral_list, blur_bilateral_list = self.unstack_filter(bilateral_filter_batch, self.d+2)
            Q_output_list = []

            for i in range(self.n_batch):

                U_tensor = U_tensor_list[i]
                splat_bilateral = splat_bilateral_list[i]
                blur_bilateral = blur_bilateral_list[i]

                Q_tensor = U_tensor

                # construct RNN part
                for j in range(self.n_unit):
                    Q_tensor_norm = tf.nn.softmax(Q_tensor)
                    Q_tensor = self.RNN_unit(Q_tensor_norm, U_tensor, spatial_weight, bilateral_weight,
                                             compatible_weight, splat_spatial, blur_spatial, splat_bilateral, blur_bilateral)

                Q_output_list.append(Q_tensor)

            # stack back to batch
            self.output = tf.stack(Q_output_list, axis=0)

            return self.output

    def unstack_filter(self, filter_batch, d):
        """
        preprocess the filter matrices from batch (has to create placeholder for each sample in batch)
        :param filter_batch:
        :return:
        """
        splat_list = filter_batch['splat']
        weight_list = filter_batch['weight']
        n_lattice_list = filter_batch['n_lattice']
        blur_list = filter_batch['blur']

        # splat_list = tf.unstack(splat_batch, num=self.n_batch)
        # weight_list = tf.unstack(weight_batch, num=self.n_batch)
        # n_lattice_list = tf.unstack(n_lattice_batch, num=self.n_batch)
        # blur_list = tf.unstack(blur_batch, num=self.n_batch)

        splat_sparse_list = []
        blur_sparse_list = []

        for i in range(self.n_batch):
            splat_sparse_matrix, blur_matrices = self.transformation_tensor(splat_list[i], weight_list[i], blur_list[i],
                                                                            n_lattice_list[i], d, n=self.n_pixel)
            splat_sparse_list.append(splat_sparse_matrix)
            blur_sparse_list.append(blur_matrices)

        return splat_sparse_list, blur_sparse_list


    def RNN_unit(self, Q_tensor, U_tensor, spatial_weight, bilateral_weight, compatible_weight_tensor,
                 spatial_splat_matrix, spatial_blur_matrices, bilateral_splat_matrix, bilateral_blur_matrices):
        """
        Construct the RNN unit of CRF
        :param Q_tensor:
        :param U_tensor:
        :param filter_weight_tensor:
        :param compatible_weight_tensor:
        :param gauss_tensor_dict:
        :param bilateral_tensor_dict:
        :return:
        """

        Q_til_spatial = self.gauss_filter(Q_tensor, spatial_splat_matrix, spatial_blur_matrices)
        Q_til_bilateral = self.gauss_filter(Q_tensor, bilateral_splat_matrix, bilateral_blur_matrices)

        Q_til_weighted = tf.multiply(Q_til_spatial, spatial_weight) + tf.multiply(Q_til_bilateral, bilateral_weight)

        Q_hat = tf.matmul(Q_til_weighted, compatible_weight_tensor)

        Q_check = U_tensor + Q_hat # add or subtract?

        # Q_check_norm = tf.nn.softmax(Q_check)

        return Q_check

    def transformation_tensor(self, splat_vector_tensor, weight_vector_tensor, blur_vector_tensor,
                              n_lattice_tensor, d, n):
        """
        Construct the transformation tensors of Gauss filtering using sparse tensor
        :param splat_vector_tensor:
        :param weight_vector_tensor:
        :param blur_vector_tensor:
        :param n_lattice_tensor:
        :return:
        """
        # first construct the indices of splat matrix from splat_vector_tensor
        ind_tmp = np.repeat(np.arange(n), (d+1))
        ind_splat = tf.stack([splat_vector_tensor, tf.constant(ind_tmp, dtype=tf.int64)], axis=1)
        splat_shape = tf.stack([n_lattice_tensor+1, tf.constant(n, dtype=tf.int64)])
        splat_sparse_matrix = tf.sparse_reorder(tf.SparseTensor(indices=ind_splat, values=weight_vector_tensor, dense_shape=splat_shape))

        # construct the list of blur matrices
        blur_vector_tensor_reshape = tf.reshape(blur_vector_tensor, (-1, 3))
        blur_vector_tensor_split = tf.split(blur_vector_tensor_reshape, d+1)
        blur_list = []

        # calculate the blur weight
        shape_tmp = tf.stack([n_lattice_tensor, 1], axis=0)
        ones_tmp = tf.ones(shape=tf.to_int32(shape_tmp), dtype=tf.float32)
        blur_weight = tf.matmul(ones_tmp, tf.constant([[0.5, 0.25, 0.25]], shape=[1,3], dtype=tf.float32))
        blur_weight = tf.reshape(blur_weight, shape=[-1])

        for i in range(d+1):
            blur_vector_tensor_dim = blur_vector_tensor_split[i]
            ind_dim_tmp = tf.slice(blur_vector_tensor_dim, begin=[0,0], size=[-1,1])
            ind_dim_tmp_tile = tf.tile(ind_dim_tmp, [1, 3])
            ind_blur_dim = tf.stack([tf.reshape(ind_dim_tmp_tile, [-1]), tf.reshape(blur_vector_tensor_dim, [-1])], axis=1)

            #tmp = tf.stack([n_lattice_tensor])
            #blur_weight = tf.tile(tf.constant([0.5, 0.25, 0.25]), tmp)

            blur_shape = tf.stack([n_lattice_tensor+1, n_lattice_tensor+1])
            blur_list.append(tf.sparse_reorder(tf.SparseTensor(indices=ind_blur_dim, values=blur_weight, dense_shape=blur_shape)))

        return splat_sparse_matrix, blur_list

    def gauss_filter(self, Q_tensor, splat_sparse_tensor, blur_sparse_list, scaling = True):
        """
        Generate the tensor after gauss filtering of Q (softmax)
        :param Q_tensor: n_pixel * n_class
        :param splat_sparse_tensor:
        :param blur_sparse_list:
        :return:
        """
        # splat
        splatted_tensor = tf.sparse_tensor_dense_matmul(splat_sparse_tensor, Q_tensor)

        # blur
        blurred_tensor = splatted_tensor
        for i in range(len(blur_sparse_list)):
            blurred_tensor = tf.sparse_tensor_dense_matmul(blur_sparse_list[i], blurred_tensor)

        # slicing
        slice_sparse_tensor = tf.sparse_transpose(splat_sparse_tensor)
        sliced_tensor = tf.sparse_tensor_dense_matmul(slice_sparse_tensor, blurred_tensor)

        # scaling
        if scaling:
            aux_tensor = tf.ones([self.n_pixel, 1])
            splatted_tensor_aux = tf.sparse_tensor_dense_matmul(splat_sparse_tensor, aux_tensor)
            # blur
            blurred_tensor_aux = splatted_tensor_aux

            for i in range(len(blur_sparse_list)):
                blurred_tensor_aux = tf.sparse_tensor_dense_matmul(blur_sparse_list[i], blurred_tensor_aux)

            sliced_tensor_aux = tf.sparse_tensor_dense_matmul(slice_sparse_tensor, blurred_tensor_aux)
            sliced_tensor_scale = tf.div(sliced_tensor, sliced_tensor_aux)

        else:
            sliced_tensor_scale = sliced_tensor

        # subtract the original Q
        Q_til = tf.subtract(sliced_tensor_scale, Q_tensor)

        return Q_til


def bilateral_batch(img_batch, sigma_bilateral):
    """
    Calculate the bilateral transformation vectors
    for a image batch
    :param img_batch:

    :return:
    """
    n_batch = img_batch.shape[0]
    H = img_batch.shape[1]
    W = img_batch.shape[2]
    # n_channel = img_batch.shape[3]

    bilateral_dict = {}

    bilateral_dict['splat'] = []
    bilateral_dict['weight'] = []
    bilateral_dict['n_lattice'] = []
    bilateral_dict['blur'] = []

    for i in range(n_batch):

        img = img_batch[i,:,:,:]
        # first calculate the img grid
        img_grid = generate_grid(H, W)
        # stack
        img_ex = np.concatenate([img, img_grid], axis=2)

        # calculate the spatial vectors
        bilateral_splat, bilateral_weight, bilateral_blur, bilateral_n_lattice = \
            transformation_vector(img_ex, sigma_bilateral)

        bilateral_dict['splat'].append(bilateral_splat)
        bilateral_dict['weight'].append(bilateral_weight)
        bilateral_dict['n_lattice'].append(bilateral_n_lattice)
        bilateral_dict['blur'].append(bilateral_blur)

    return bilateral_dict


def spatial_global(H, W, gamma):
    """
    construct the vector for spatial filter
    :param H:
    :param W:
    :param gamma:
    :return:
    """
    spatial_dict = {}
    img_grid = generate_grid(H, W)
    splat_vector, weight_vector, blur_vector, n_lattice = transformation_vector(img_grid, gamma*np.ones(2))
    spatial_dict['splat'] = splat_vector
    spatial_dict['weight'] = weight_vector
    spatial_dict['n_lattice'] = n_lattice
    spatial_dict['blur'] = blur_vector

    return spatial_dict


def transformation_vector(img_ex, sigma):
    """
    :param img_ex: the extended image array (H * W * n_channel)
    :param sigma: the spatial deviation of filter (n_channel)
    :return: the transformation vectors: splat and list of blurs
    """

    H = img_ex.shape[0]
    W = img_ex.shape[1]
    n_channel = img_ex.shape[2]

    if len(sigma) != n_channel:
        print("invalid sigma size")
        exit(1)

    img_ex_norm = img_ex.astype('float32')/sigma

    # calculate the transformation vectors
    img_flat = img_ex_norm.flatten().astype('float32')
    pmh = permutohedral.PermutohedralLattice(d_=n_channel, nData_= H * W)
    pmh.generate_splat_vector(img_flat)
    pmh.generate_blur_vector()
    splat_vector = pmh.get_enclosing_simplices(H * W * (n_channel + 1))
    weight_vector = pmh.get_weights(H * W * (n_channel + 1))
    n_lattice = pmh.get_lattice_points()
    blur_vector = pmh.get_blur((n_channel + 1) * 3 * n_lattice)

    del pmh

    return splat_vector, weight_vector, blur_vector, n_lattice


def generate_grid (H, W):
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    img_grid = np.dstack((y, x))
    return img_grid

