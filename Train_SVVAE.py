import os
from os.path import isfile, join, splitext
import math
import numpy as np
import tensorflow as tf
import SimpleITK as sitk
from Utilities import *
import matplotlib.pyplot as plt
from GVAE import GVAE


def run_training(num_epoch, batch_size, lr):
    # setup some directory
    model_filename = 'gvae'
    model_save_dir = './ckpt/' + model_filename

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # load PANCREAS data
    srcFolder = 'Dataset/Train/'
    dat = loadImages(srcFolder)
    gt = loadGT(srcFolder)
    N = dat.shape[0]

    # build GVAE
    gvae = GVAE()
    train_step = tf.train.AdamOptimizer(lr).minimize(gvae.total_loss)

    # generate graph model for debug - Tensorboard
    # sess = tf.Session()
    # writer = tf.summary.FileWriter('logs/', sess.graph)

    # open a training session
    sess = tf.InteractiveSession()

    # initialize variables
    sess.run(tf.global_variables_initializer())

    # training
    num_iter = int(math.ceil(N / batch_size))
    print('Starting training ... %d iteration per epoch' % num_iter)

    for i in range(num_epoch):
        rand_idx = np.random.permutation(N)

        for it in range(num_iter):
            idx = rand_idx[it*batch_size:(it+1)*batch_size]
            X_batch = dat[idx, :, :, :, :]
            _, total_loss = sess.run([train_step, gvae.total_loss], feed_dict={gvae.x: X_batch})
            print("Iter [%d/%d] loss=%.6f" % (it, num_iter, total_loss))

        # print current epoch error
        # _, total_loss = sess.run([gvae.x[rand_idx[0]], gvae.total_loss], feed_dict={gvae.x: dat[0:1, :, :, :, :]})
        print("Epoch [%d/%d] total_loss=%.6f" % (i + 1, num_epoch, total_loss))

        # save model
        if (i+1) % 500 == 0:
            saver = tf.train.Saver(max_to_keep=10)
            saver.save(sess, os.path.join(model_save_dir, model_filename, str(i)))


if __name__ == '__main__':
    num_epoch = 10000000
    batch_size = 2
    lr = 1e-2
    run_training(num_epoch, batch_size, lr)
