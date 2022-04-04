# -*- coding:utf-8 -*-
import tensorflow as tf

def LiCENt_(input_shape=(192, 192, 1)):

    h = inputs = tf.keras.Input(input_shape)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=2)(h)
    h = tf.keras.layers.ReLU()(h)
    skip1 = h

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=2)(h)
    h = tf.keras.layers.ReLU()(h)
    skip2 = h

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=2)(h)
    h = tf.keras.layers.ReLU()(h)
    skip3 = h

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=2)(h)
    h = tf.keras.layers.ReLU()(h)
    skip4 = h

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=2)(h)
    h = tf.keras.layers.ReLU()(h)
    skip5 = h

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=2)(h)
    h = tf.keras.layers.ReLU()(h)
    skip6 = h

    h = tf.keras.layers.Conv2D(filters=16, kernel_size=3)(h)
    h = tf.keras.layers.ReLU()(h)

    #############################################################################################

    h = tf.image.resize(h, [3,3])
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1)(h)
    h = skip6 + h

    h = tf.image.resize(h, [6,6])
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1)(h)
    h = skip5 + h

    h = tf.image.resize(h, [12,12])
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1)(h)
    h = skip4 + h

    h = tf.image.resize(h, [24,24])
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1)(h)
    h = skip3 + h

    h = tf.image.resize(h, [48,48])
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1)(h)
    h = skip2 + h

    h = tf.image.resize(h, [96,96])
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1)(h)
    h = skip1 + h

    h = tf.image.resize(h, [192,192])
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1)(h)
    h = inputs + h

    #############################################################################################

    h = tf.concat([inputs, h], -1)
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1)(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1)(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1)(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1)(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1)(h)


    return tf.keras.Model(inputs=inputs, outputs=h)
