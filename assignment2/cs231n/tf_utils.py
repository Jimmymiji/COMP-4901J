import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt

def conv_bn_relu_pool_unit(input,is_training,num_filters,kernel_size,pool_size,
                            use_stride,use_pool,dropout_rate):
    if not use_stride:
        conv = tf.layers.conv2d(input,num_filters,kernel_size,padding='same')
    else:
        conv = tf.layers.conv2d(input,num_filters,kernel_size,padding='same',
                            activation=None,kernel_initializer=initializer,
                            strides=[2,2])
    bn = tf.layers.batch_normalization(conv,training=is_training)
    dropout = tf.layers.dropout(bn,dropout_rate,
                                [tf.shape(bn)[0],1,1,num_filters],
                                training=is_training)
    relu = tf.nn.relu(dropout)
    if use_pool:
        out = tf.layers.max_pooling2d(relu,pool_size=[2,2],strides=[2,2])
    else:
        out = relu
    return out

def conv_bn_relu_conv_bn_relu_pool_unit(input,is_training,num_filters,kernel_size,pool_size,
                                        use_stride,use_pool,dropout_rate):
    if not use_stride:
        conv1 = tf.layers.conv2d(input,num_filters,kernel_size,padding='same')
    else:
        conv1 = tf.layers.conv2d(input,num_filters,kernel_size,padding='same',strides=[2,2])
    bn1 = tf.layers.batch_normalization(conv1,training=is_training)
    dropout1 = tf.layers.dropout(bn1,dropout_rate,
                                [tf.shape(bn1)[0],1,1,num_filters],
                                training=is_training)
    relu1 = tf.nn.relu(dropout1)
    if not use_stride:
        conv2 = tf.layers.conv2d(relu1,num_filters,kernel_size,padding='same')
    else:
        conv2 = tf.layers.conv2d(relu1,num_filters,kernel_size,padding='same',strides=[2,2])
    bn2 = tf.layers.batch_normalization(conv2,training=is_training)
    dropout2 = tf.layers.dropout(bn2,dropout_rate,
                                [tf.shape(bn2)[0],1,1,num_filters],
                                training=is_training)
    relu2 = tf.nn.relu(dropout2)
    if use_pool:
        out = tf.layers.max_pooling2d(relu2,pool_size=[2,2],strides=[2,2])
    else:
        out = relu2
    return out
