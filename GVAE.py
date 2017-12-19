import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim import fully_connected as fc

DTYPE = tf.float32
latent_sz = 256

class GVAE:

    def __init__(self):
        self._build_graph_()

    def _weight_variable(self, name, shape):
        return tf.get_variable(name, shape, DTYPE, tf.truncated_normal_initializer(stddev=0.1))

    def _bias_variable(self, name, shape):
        return tf.get_variable(name, shape, DTYPE, tf.constant_initializer(0, dtype=DTYPE))

    def _build_graph_(self):
        with tf.name_scope('inputs'):
            x = tf.placeholder(tf.float32, [None, 128, 128, 64, 1], name='x_input')
            input_layer = x

        ### ENCODER (parametrization of approximate posterior q(z|x))
        with tf.variable_scope('level1_EC', reuse=None) as scope:  # LEVEL 1 for Encoder
            prev_layer = input_layer

            # replicate the volume (generate 128x128x64 with x16)
            d = prev_layer
            tile1_l1 = tf.tile(d, [1, 1, 1, 1, 16])

            # conv3d1 (generate 128x128x64 with x16)
            in_filters = 1
            out_filters = 16
            kernel = self._weight_variable('weights1_l1', [5, 5, 5, in_filters, out_filters])
            conv1_l1 = tf.nn.conv3d(prev_layer, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
            biases = self._bias_variable('biases1_l1', [out_filters])
            conv1_l1_out = tf.nn.bias_add(conv1_l1, biases)

            # element-wise sum tile1_l1 & conv1_l1_out (optional: residual output to decode part)
            sum1_l1 = tf.add(tile1_l1, conv1_l1_out)
            level1_out = tf.nn.relu(sum1_l1, name=scope.name)

            # Down conv for level 2 (generate 64x64x32 with x32)
            in_filters = 16
            out_filters = 32
            kernel = self._weight_variable('weights2_l1', [2, 2, 2, in_filters, out_filters])
            conv2_l1 = tf.nn.conv3d(level1_out, kernel, strides=[1, 2, 2, 2, 1], padding='SAME')
            biases = self._bias_variable('biases2_l1', [out_filters])
            conv2_l1_out = tf.nn.bias_add(conv2_l1, biases)
            level2_in = tf.nn.relu(conv2_l1_out, name=scope.name)

        with tf.variable_scope('level2_EC') as scope:  # LEVEL 2 for Encoder
            prev_layer = level2_in

            # conv3d1 (generate 64x64x32 with x32)
            in_filters = 32
            out_filters = 32
            kernel = self._weight_variable('weights1_l2', [5, 5, 5, in_filters, out_filters])
            conv1_l2 = tf.nn.conv3d(prev_layer, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
            biases = self._bias_variable('biases1_l2', [out_filters])
            conv1_l2_out = tf.nn.bias_add(conv1_l2, biases)

            # element-wise sum level2_in & conv2_l2_out (optional: residual output to decode part)
            sum1_l2 = tf.add(level2_in, conv1_l2_out)
            level2_out = tf.nn.relu(sum1_l2, name=scope.name)

            # Down conv for level 3 (generate 32x32x16 with x64)
            in_filters = 32
            out_filters = 64
            kernel = self._weight_variable('weights2_l2', [2, 2, 2, in_filters, out_filters])
            conv2_l2 = tf.nn.conv3d(level2_out, kernel, strides=[1, 2, 2, 2, 1], padding='SAME')
            biases = self._bias_variable('biases2_l2', [out_filters])
            conv2_l2_out = tf.nn.bias_add(conv2_l2, biases)
            level3_in = tf.nn.relu(conv2_l2_out, name=scope.name)

        with tf.variable_scope('level3_EC') as scope:  # LEVEL 3 for Encoder
            prev_layer = level3_in

            # conv3d1 (generate 32x32x16 with x64)
            in_filters = 64
            out_filters = 64
            kernel = self._weight_variable('weights1_l3', [5, 5, 5, in_filters, out_filters])
            conv1_l3 = tf.nn.conv3d(prev_layer, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
            biases = self._bias_variable('biases1_l3', [out_filters])
            conv1_l3_out = tf.nn.bias_add(conv1_l3, biases)

            # conv3d2 (generate 32x32x16 with x64)
            in_filters = 64
            out_filters = 64
            kernel = self._weight_variable('weights2_l3', [5, 5, 5, in_filters, out_filters])
            conv2_l3 = tf.nn.conv3d(conv1_l3_out, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
            biases = self._bias_variable('biases2_l3', [out_filters])
            conv2_l3_out = tf.nn.bias_add(conv2_l3, biases)

            # element-wise sum level3_in & conv2_l3_out (optional: residual output to decode part)
            sum1_l3 = tf.add(level3_in, conv2_l3_out)
            level3_out = tf.nn.relu(sum1_l3, name=scope.name)

            # Down conv for level 4 (generate 16x16x8 with x128)
            in_filters = 64
            out_filters = 128
            kernel = self._weight_variable('weights3_l3', [2, 2, 2, in_filters, out_filters])
            conv3_l3 = tf.nn.conv3d(level3_out, kernel, strides=[1, 2, 2, 2, 1], padding='SAME')
            biases = self._bias_variable('biases3_l3', [out_filters])
            conv3_l3_out = tf.nn.bias_add(conv3_l3, biases)
            level4_in = tf.nn.relu(conv3_l3_out, name=scope.name)

        with tf.variable_scope('level4_EC') as scope:  # LEVEL 4 for Encoder
            prev_layer = level4_in

            # conv3d1 (generate 16x16x8 with x128)
            in_filters = 128
            out_filters = 128
            kernel = self._weight_variable('weights1_l4', [5, 5, 5, in_filters, out_filters])
            conv1_l4 = tf.nn.conv3d(prev_layer, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
            biases = self._bias_variable('biases1_l4', [out_filters])
            conv1_l4_out = tf.nn.bias_add(conv1_l4, biases)

            # conv3d2 (generate 16x16x8 with x128)
            in_filters = 128
            out_filters = 128
            kernel = self._weight_variable('weights2_l4', [5, 5, 5, in_filters, out_filters])
            conv2_l4 = tf.nn.conv3d(conv1_l4_out, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
            biases = self._bias_variable('biases2_l4', [out_filters])
            conv2_l4_out = tf.nn.bias_add(conv2_l4, biases)

            # element-wise sum level4_in & conv2_l4_out (optional: residual output to decode part)
            sum1_l4 = tf.add(level4_in, conv2_l4_out)
            level4_out = tf.nn.relu(sum1_l4, name=scope.name)

            # Down conv for level 5 (generate 8x8x4 with x256)
            in_filters = 128
            out_filters = 256
            kernel = self._weight_variable('weights3_l4', [2, 2, 2, in_filters, out_filters])
            conv3_l4 = tf.nn.conv3d(level4_out, kernel, strides=[1, 2, 2, 2, 1], padding='SAME')
            biases = self._bias_variable('biases3_l4', [out_filters])
            conv3_l4_out = tf.nn.bias_add(conv3_l4, biases)
            level5_in = tf.nn.relu(conv3_l4_out, name=scope.name)

        with tf.variable_scope('level5_EC') as scope:  # LEVEL 5 for Encoder
            prev_layer = level5_in

            # conv3d1 (generate 8x8x4 with x256)
            in_filters = 256
            out_filters = 256
            kernel = self._weight_variable('weights1_l5', [5, 5, 5, in_filters, out_filters])
            conv1_l5 = tf.nn.conv3d(prev_layer, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
            biases = self._bias_variable('biases1_l5', [out_filters])
            conv1_l5_out = tf.nn.bias_add(conv1_l5, biases)

            # conv3d2 (generate 8x8x4 with x256)
            in_filters = 256
            out_filters = 256
            kernel = self._weight_variable('weights2_l5', [5, 5, 5, in_filters, out_filters])
            conv2_l5 = tf.nn.conv3d(conv1_l5_out, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
            biases = self._bias_variable('biases2_l5', [out_filters])
            conv2_l5_out = tf.nn.bias_add(conv2_l5, biases)

            # element-wise sum level5_in & conv2_l5_out (optional: residual output to decode part)
            sum1_l5 = tf.add(level5_in, conv2_l5_out)
            level5_out = tf.nn.relu(sum1_l5, name=scope.name)

        ## fully connected layer for level5_out(8x8x4 with x256 -> 1x65536 -> 1x128)
        with tf.variable_scope('fc_EC') as scope:  # LEVEL 5 for Encoder
            level5_out = tf.contrib.layers.flatten(level5_out)
            z_param = fc(level5_out, latent_sz * 2, activation_fn=None)

            z_log_sigma_sq = z_param[:, :latent_sz]  # log deviation square of q(z|x)
            z_mu = z_param[:, latent_sz:]  # mean of q(z|x)

            # sample latent variable z from q(z|x)
            eps = tf.random_normal(shape=tf.shape(z_log_sigma_sq))
            z = tf.sqrt(tf.exp(z_log_sigma_sq)) * eps + z_mu

        ### DECODER (mirror structure of the encoder)
        ## fully connected layer for level5 in (1x64 -> 1x65536 -> 8x8x4 with x256)
        with tf.variable_scope('latent_to_3D') as scope:  # latent space to 3D
            level5_in = fc(z, 65536, activation_fn=None)
            level5_in = tf.reshape(level5_in, (tf.shape(level5_in)[0], 8, 8, 4, 256))

        with tf.variable_scope('level5_DC') as scope:   # LEVEL 5 for Decoder
            prev_layer = level5_in

            # conv3d1 (generate 8x8x4 with x256)
            in_filters = 256
            out_filters = 256
            kernel = self._weight_variable('weights1_l5', [5, 5, 5, in_filters, out_filters])
            conv1_l5 = tf.nn.conv3d(prev_layer, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
            biases = self._bias_variable('biases1_l5', [out_filters])
            conv1_l5_out = tf.nn.bias_add(conv1_l5, biases)

            # conv3d2 (generate 8x8x4 with x256)
            in_filters = 256
            out_filters = 256
            kernel = self._weight_variable('weights2_l5', [5, 5, 5, in_filters, out_filters])
            conv2_l5 = tf.nn.conv3d(conv1_l5_out, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
            biases = self._bias_variable('biases2_l5', [out_filters])
            conv2_l5_out = tf.nn.bias_add(conv2_l5, biases)

            # element-wise sum conv1_lat23d & conv2_l5_out
            sum1_l5 = tf.add(level5_in, conv2_l5_out)
            level5_out = tf.nn.relu(sum1_l5, name=scope.name)

            # Up conv for level 4 (generate 16x16x8 with x128)
            in_filters = 256
            out_filters = 128
            out_shape = [tf.shape(level5_out)[0], 16, 16, 8, 128]
            kernel = self._weight_variable('weights3_l5', [2, 2, 2, out_filters, in_filters])
            conv3_l5 = tf.nn.conv3d_transpose(level5_out, kernel, output_shape=out_shape, strides=[1, 2, 2, 2, 1], padding='SAME')
            biases = self._bias_variable('biases3_l5', [out_filters])
            conv3_l5_out = tf.nn.bias_add(conv3_l5, biases)
            level4_in = tf.nn.relu(conv3_l5_out, name=scope.name)

        with tf.variable_scope('level4_DC') as scope:   # LEVEL 4 for Decoder
            prev_layer = level4_in

            # conv3d1 (generate 16x16x8 with x128)
            in_filters = 128
            out_filters = 128
            kernel = self._weight_variable('weights1_l4', [5, 5, 5, in_filters, out_filters])
            conv1_l4 = tf.nn.conv3d(prev_layer, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
            biases = self._bias_variable('biases1_l4', [out_filters])
            conv1_l4_out = tf.nn.bias_add(conv1_l4, biases)

            # conv3d2 (generate 16x16x8 with x128)
            in_filters = 128
            out_filters = 128
            kernel = self._weight_variable('weights2_l4', [5, 5, 5, in_filters, out_filters])
            conv2_l4 = tf.nn.conv3d(conv1_l4_out, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
            biases = self._bias_variable('biases2_l4', [out_filters])
            conv2_l4_out = tf.nn.bias_add(conv2_l4, biases)

            # element-wise sum level4_in & conv2_l4_out
            sum1_l4 = tf.add(level4_in, conv2_l4_out)
            level4_out = tf.nn.relu(sum1_l4, name=scope.name)

            # Up conv for level 3 (generate 32x32x16 with x64)
            in_filters = 128
            out_filters = 64
            out_shape = [tf.shape(level4_out)[0], 32, 32, 16, 64]
            kernel = self._weight_variable('weights3_l4', [2, 2, 2, out_filters, in_filters])
            conv3_l4 = tf.nn.conv3d_transpose(level4_out, kernel, output_shape=out_shape, strides=[1, 2, 2, 2, 1], padding='SAME')
            biases = self._bias_variable('biases3_l4', [out_filters])
            conv3_l4_out = tf.nn.bias_add(conv3_l4, biases)
            level3_in = tf.nn.relu(conv3_l4_out, name=scope.name)

        with tf.variable_scope('level3_DC') as scope:  # LEVEL 3 for Decoder
            prev_layer = level3_in

            # conv3d1 (generate 32x32x16 with x64)
            in_filters = 64
            out_filters = 64
            kernel = self._weight_variable('weights1_l3', [5, 5, 5, in_filters, out_filters])
            conv1_l3 = tf.nn.conv3d(prev_layer, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
            biases = self._bias_variable('biases1_l3', [out_filters])
            conv1_l3_out = tf.nn.bias_add(conv1_l3, biases)

            # conv3d2 (generate 32x32x16 with x64)
            in_filters = 64
            out_filters = 64
            kernel = self._weight_variable('weights2_l3', [5, 5, 5, in_filters, out_filters])
            conv2_l3 = tf.nn.conv3d(conv1_l3_out, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
            biases = self._bias_variable('biases2_l3', [out_filters])
            conv2_l3_out = tf.nn.bias_add(conv2_l3, biases)

            # element-wise sum level3_in & conv2_l3_out
            sum1_l3 = tf.add(level3_in, conv2_l3_out)
            level3_out = tf.nn.relu(sum1_l3, name=scope.name)

            # Up conv for level 2 (generate 64x64x32 with x32)
            in_filters = 64
            out_filters = 32
            out_shape = [tf.shape(level3_out)[0], 64, 64, 32, 32]
            kernel = self._weight_variable('weights3_l3', [2, 2, 2, out_filters, in_filters])
            conv3_l3 = tf.nn.conv3d_transpose(level3_out, kernel, output_shape=out_shape, strides=[1, 2, 2, 2, 1], padding='SAME')
            biases = self._bias_variable('biases3_l3', [out_filters])
            conv3_l3_out = tf.nn.bias_add(conv3_l3, biases)
            level2_in = tf.nn.relu(conv3_l3_out, name=scope.name)

        with tf.variable_scope('level2_DC') as scope:  # LEVEL 2 for Decoder
            prev_layer = level2_in

            # conv3d1 (generate 64x64x32 with x32)
            in_filters = 32
            out_filters = 32
            kernel = self._weight_variable('weights1_l2', [5, 5, 5, in_filters, out_filters])
            conv1_l2 = tf.nn.conv3d(prev_layer, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
            biases = self._bias_variable('biases1_l2', [out_filters])
            conv1_l2_out = tf.nn.bias_add(conv1_l2, biases)

            # element-wise sum level2_in & conv1_l2_out
            sum1_l2 = tf.add(level2_in, conv1_l2_out)
            level2_out = tf.nn.relu(sum1_l2, name=scope.name)

            # Up conv for level 1 (generate 128x128x64 with x16)
            in_filters = 32
            out_filters = 16
            out_shape = [tf.shape(level2_out)[0], 128, 128, 64, 16]
            kernel = self._weight_variable('weights2_l2', [2, 2, 2, out_filters, in_filters])
            conv2_l2 = tf.nn.conv3d_transpose(level2_out, kernel, output_shape=out_shape, strides=[1, 2, 2, 2, 1], padding='SAME')
            biases = self._bias_variable('biases2_l2', [out_filters])
            conv2_l2_out = tf.nn.bias_add(conv2_l2, biases)
            level1_in = tf.nn.relu(conv2_l2_out, name=scope.name)

        with tf.variable_scope('level1_DC') as scope:  # LEVEL 1 for Decoder
            prev_layer = level1_in

            # conv3d1 (generate 128x128x64 with x16)
            in_filters = 16
            out_filters = 16
            kernel = self._weight_variable('weights1_l1', [5, 5, 5, in_filters, out_filters])
            conv1_l1 = tf.nn.conv3d(prev_layer, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
            biases = self._bias_variable('biases1_l1', [out_filters])
            conv1_l1_out = tf.nn.bias_add(conv1_l1, biases)

            # element-wise sum level1_in & conv1_l1_out
            sum1_l1 = tf.add(level1_in, conv1_l1_out)
            level1_out = tf.nn.relu(sum1_l1, name=scope.name)

            # Up conv for final output (generate 128x128x64 with x1)
            in_filters = 16
            out_filters = 1
            out_shape = [tf.shape(level1_out)[0], 128, 128, 64, 1]
            kernel = self._weight_variable('weights2_l1', [2, 2, 2, out_filters, in_filters])
            conv2_l1 = tf.nn.conv3d_transpose(level1_out, kernel, output_shape=out_shape, strides=[1, 1, 1, 1, 1], padding='SAME')
            biases = self._bias_variable('biases2_l1', [out_filters])
            x_recon = tf.nn.bias_add(conv2_l1, biases)

        # loss: negative of Evidence Lower BOund (ELBO)
        # 1. KL-divergence: KL(q(z|x)||p(z))
        # (divergence between two multi-variate normal distribution, please refer to Wiki)
        kl_loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.exp(z_log_sigma_sq) + tf.square(z_mu) - 1 - z_log_sigma_sq, axis=1))

        # 2. Likelihood: p(x|z)
        # also called as reconstruction loss
        # we parametrized it with binary cross-entropy loss as MNIST contains binary images
        eps = 1e-10  # add small number to avoid log(0.0)
        recon_loss = tf.reduce_mean(-tf.reduce_sum(x * tf.log(eps + x_recon) + (1 - x) * tf.log(1 - x_recon + eps), axis=1))
        total_loss = kl_loss + recon_loss

        self.z = z
        self.total_loss, self.recon_loss, self.kl_loss = total_loss, recon_loss, kl_loss
        self.x = x
        self.x_recon = x_recon
