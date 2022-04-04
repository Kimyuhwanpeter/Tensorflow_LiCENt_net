# -*- coding:utf-8 -*-
from LiCENt_model import *

import matplotlib.pyplot as plt
import numpy as np
import easydict
import cv2

FLAGS = easydict.EasyDict({"img_size": 192,

                           "tr_img_path": "D:/[1]DB/[5]4th_paper_DB/crop_weed/datasets_IJRR2017/low_light2/",

                           "tr_lab_path": "D:/[1]DB/[5]4th_paper_DB/crop_weed/datasets_IJRR2017/raw_aug_rgb_img/",
                           
                           "tr_txt_path": "D:/[1]DB/[5]4th_paper_DB/crop_weed/datasets_IJRR2017/train.txt",
                           
                           "batch_size": 4,
                           
                           "epochs": 50,
                           
                           "lr": 0.0001,

                           "train": True,

                           "sample_images": "C:/Users/Yuhwan/Downloads/sample_images",
                           
                           "save_checkpoint": "C:/Users/Yuhwan/Downloads/checkpoint",

                           "pre_checkpoint": False,

                           "pre_checkpoint_path": "",
                           
                           "te_img_path": "D:/[1]DB/[5]4th_paper_DB/crop_weed/CropWeed Field Image Dataset (CWFID)/dataset-1.0/low_light/",
                           
                           "test_images": "D:/[1]DB/[5]4th_paper_DB/crop_weed/CropWeed Field Image Dataset (CWFID)/dataset-1.0/restored_low_light_DSLR"})

# LiCENt: Low-light image enhancement using the light channel of HSL
optim = tf.keras.optimizers.Adam(FLAGS.lr)

def tr_func(img_list, lab_list):

    img = tf.io.read_file(img_list)
    img = tf.image.decode_png(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size]) / 255

    # HSV, HSL ==> H is same

    lab = tf.io.read_file(lab_list)
    lab = tf.image.decode_png(lab, 3)
    lab = tf.image.resize(lab, [FLAGS.img_size, FLAGS.img_size]) / 255

    return img, lab

def te_func(img_data):

    img = tf.io.read_file(img_data)
    img = tf.image.decode_png(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size]) / 255

    return img

def SSIM_loss(y_true, y_pred):

    loss = 0
    #for i in range(FLAGS.batch_size):

    #    #y_true_ = tf.reshape(y_true[i], [FLAGS.img_size*FLAGS.img_size])
    #    #y_pred_ = tf.reshape(y_pred[i], [FLAGS.img_size*FLAGS.img_size])
    #    y_true_ = tf.cast(y_true[i, :, :, 0], tf.float32)
    #    y_pred_ = tf.cast(y_pred[i, :, :, 0], tf.float32)

    #    y_true_var = tf.math.reduce_variance(y_true_, -1)
    #    y_pred_var = tf.math.reduce_variance(y_pred_, -1)
    #    y_true_mean = tf.reduce_mean(y_true_, -1)
    #    y_pred_mean = tf.reduce_mean(y_pred_, -1)
    #    covar = np.cov(y_true_.numpy(), y_pred_.numpy())[0][1]

    #    #ssim = (2*y_pred_mean*y_true_mean + 0.0001) * (2*covar + 0.0009) / (tf.pow(y_pred_mean, 2.0) + tf.pow(y_true_mean, 2.0) + 0.0001) * (y_pred_var + y_true_var + 0.0009)
    #    ssim = tf.image.ssim(y_true_, y_pred_, max_val=1.0)
    #    print(ssim)
    #    loss += 1 - ssim

    #loss /= FLAGS.batch_size

    ssim_loss = tf.reduce_mean(1. - tf.image.ssim(y_true, y_pred, max_val=1.0, k1=0.0001, k2=0.0009))

    return ssim_loss

#@tf.function
def cal_loss(model, images, labels):

    with tf.GradientTape() as tape:

        output = model(images, True)

        loss = SSIM_loss(labels, output)

    grads = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

    return loss

def main():
    
    model = LiCENt_(input_shape=(FLAGS.img_size, FLAGS.img_size, 1))
    model.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(model=model, optim=optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored!!!")

    if FLAGS.train:
        count = 0;

        data_list = np.loadtxt(FLAGS.tr_txt_path, dtype="<U200", skiprows=0, usecols=0)
        img_data = [FLAGS.tr_img_path + data for data in data_list]
        img_data = np.array(img_data)
        lab_data = [FLAGS.tr_lab_path + data for data in data_list]
        lab_data = np.array(lab_data)

        for epoch in range(FLAGS.epochs):

            tr_gener = tf.data.Dataset.from_tensor_slices((img_data, lab_data))
            tr_gener = tr_gener.shuffle(len(img_data))
            tr_gener = tr_gener.map(tr_func)
            tr_gener = tr_gener.batch(FLAGS.batch_size)
            tr_gener = tr_gener.prefetch(tf.data.experimental.AUTOTUNE)

            tr_iter = iter(tr_gener)
            tr_idx = len(img_data) // FLAGS.batch_size

            for step in range(tr_idx):
                batch_images, batch_labels = next(tr_iter)

                batch_image_buf = []
                batch_label_buf = []
                for i in range(FLAGS.batch_size):
                    batch_image = cv2.cvtColor(batch_images[i].numpy(), cv2.COLOR_RGB2HLS)
                    
                    L = batch_image[:, :, 1]
                    L = L[:, :, np.newaxis]
                    batch_image_buf.append(L)

                    batch_label = cv2.cvtColor(batch_labels[i].numpy(), cv2.COLOR_RGB2HLS)

                    L_ = batch_label[:, :, 1]
                    L_ = L_[:, :, np.newaxis]
                    batch_label_buf.append(L_)

                batch_image_buf = np.array(batch_image_buf)
                batch_label_buf = np.array(batch_label_buf)

                loss = cal_loss(model, batch_image_buf, batch_label_buf)

                if count % 10 == 0:
                    print("Epoch: {} ssim loss = {} [{}/{}]".format(epoch, loss, step+1, tr_idx))

                if count % 100 == 0:

                    for i in range(FLAGS.batch_size):
                        batch_image = cv2.cvtColor(batch_images[i].numpy(), cv2.COLOR_RGB2HLS)
                        H = batch_image[:, :, 0]
                        L = batch_image[:, :, 1]
                        S = batch_image[:, :, 2]

                        H = H[:, :, np.newaxis]
                        L = L[np.newaxis, :, :, np.newaxis]
                        S = S[:, :, np.newaxis]

                        output = model(L, False)
                        output = output[0]
                        
                        final_image = tf.concat([H, output, S], -1).numpy()
                        rgb_final_image = cv2.cvtColor(final_image, cv2.COLOR_HLS2RGB)
                        rgb_final_image = np.array(rgb_final_image)
                        plt.imsave(FLAGS.sample_images + "/{}_predict_{}.png".format(count, i), rgb_final_image)
                        plt.imsave(FLAGS.sample_images + "/{}_label_{}.png".format(count, i), batch_labels[i])

                count += 1

            ckpt = tf.train.Checkpoint(model=model, optim=optim)
            ckpt.save(FLAGS.save_checkpoint + "/" + "LiCENt_Net.ckpt")

    else:
        data_list = os.listdir(FLAGS.te_img_path)
        img_data = [FLAGS.te_img_path + data for data in data_list]
        img_data = np.array(img_data)

        te_gener = tf.data.Dataset.from_tensor_slices(img_data)
        te_gener = te_gener.map(te_func)
        te_gener = te_gener.batch(1)
        te_gener = te_gener.prefetch(tf.data.experimental.AUTOTUNE)

        te_iter = iter(te_gener)
        te_idx = len(img_data) // 1
        for step in range(te_idx):
            print("Saving restored images....{}".format(step + 1))
            images = next(te_iter)

            image = cv2.cvtColor(images[0].numpy(), cv2.COLOR_RGB2HLS)
            H = image[:, :, 0]
            L = image[:, :, 1]
            S = image[:, :, 2]

            H = H[:, :, np.newaxis]
            L = L[np.newaxis, :, :, np.newaxis]
            S = S[:, :, np.newaxis]

            output = model(L, False)
            output = output[0]
                        
            final_image = tf.concat([H, output, S], -1).numpy()
            rgb_final_image = cv2.cvtColor(final_image, cv2.COLOR_HLS2RGB)
            rgb_final_image = np.array(rgb_final_image)
            plt.imsave(FLAGS.test_images + "/{}".format(name), rgb_final_image)



if __name__ == "__main__":
    main()
