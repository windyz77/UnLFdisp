# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

"""Fully convolutional model for monocular depth estimation
    by Clement Godard, Oisin Mac Aodha and Gabriel J. Brostow
    http://visual.cs.ucl.ac.uk/pubs/monoDepth/
"""

from __future__ import absolute_import, division, print_function
from collections import namedtuple

import tensorflow.contrib.slim as slim

from mat_real_scene import *
from bilinear_sampler_4_old import *
from optical_flow_warp_fwd import *

monodepth_parameters = namedtuple('parameters',
                                  'height, width, '
                                  'batch_size, '
                                  'num_threads, '
                                  'num_epochs, '
                                  'use_deconv, '
                                  'alpha_image_loss, '
                                  'dp_consistency_sigmoid_scale, '
                                  'disp_gradient_loss_weight, '
                                  'centerSymmetry_loss_weight, '
                                  'disp_consistency_loss_weight, '
                                  'full_summary')

"""monodepth LF model"""


class MonodepthModel(object):

    def __init__(self, params, mode, images_list, reuse_variables=None, model_index=None):

        self.params = params
        self.mode = mode
        self.model_collection = ['model_' + str(model_index)]

        # if self.flag:
        #
        #     self.flag = 2
        self.bz = params.batch_size
        self.images_list = images_list
        self.center = self.images_list[0]

        self.reuse_variables = reuse_variables

        self.build_model()
        self.build_outputs()

        if self.mode == 'test':
            return
        elif self.mode == 'test_flip_up_lr':
            return

        self.build_losses()
        self.build_summaries()

    def gradient_x(self, img):
        gx = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gx

    def gradient_y(self, img):
        gy = img[:, :-1, :, :] - img[:, 1:, :, :]
        return gy

    def upsample_nn(self, x, ratio):
        s = tf.shape(x)
        h = s[1]
        w = s[2]
        return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])

    def generate_image_left(self, img, disp):
        return bilinear_sampler_1d_h(img, -disp)

    def generate_image_right(self, img, disp):
        return bilinear_sampler_1d_h(img, disp)

    def generate_image_top(self, img, disp):
        return bilinear_sampler_1d_v(img, -disp)

    def generate_image_bottom(self, img, disp):
        return bilinear_sampler_1d_v(img, disp)

    def generate_image_topleft(self, img, disp_x, disp_y):
        return bilinear_sampler_2d(img, -disp_x, -disp_y)

    def generate_image_topright(self, img, disp_x, disp_y):
        return bilinear_sampler_2d(img, disp_x, -disp_y)

    def generate_image_bottomleft(self, img, disp_x, disp_y):
        return bilinear_sampler_2d(img, -disp_x, disp_y)

    def generate_image_bottomright(self, img, disp_x, disp_y):
        return bilinear_sampler_2d(img, disp_x, disp_y)


    def SSIM_no_mask(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 3, 1, 'SAME')
        mu_y = slim.avg_pool2d(y, 3, 1, 'SAME')

        sigma_x = slim.avg_pool2d(x ** 2, 3, 1, 'SAME') - mu_x ** 2
        sigma_y = slim.avg_pool2d(y ** 2, 3, 1, 'SAME') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y, 3, 1, 'SAME') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    def get_disparity_smoothness(self, disp, img):
        disp_gradients_x = self.gradient_x(disp)
        disp_gradients_y = self.gradient_y(disp)

        image_gradients_x = self.gradient_x(img)
        image_gradients_y = self.gradient_y(img)

        weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_x), 3, keepdims=True))
        weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_y), 3, keepdims=True))

        smoothness_x = tf.abs(disp_gradients_x * weights_x)
        smoothness_y = tf.abs(disp_gradients_y * weights_y)
        return tf.reduce_mean(smoothness_x) + tf.reduce_mean(smoothness_y)

    # it's checked with geonet
    def get_disp(self, x):
        disp = (self.conv(x, 25, 3, 1, activation_fn=tf.nn.sigmoid) - 0.5) * 8
        return disp

    def get_centerdisp(self, x):
        # 9 out, the range of sigmoid is (0,1), however, the range of disparity is (-4,4)
        # disp = (self.conv(x, 9, 3, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * 8
        disp = (self.conv(x, 1, 3, 1, activation_fn=tf.nn.sigmoid) - 0.5) * 8
        return disp

    """
    # it's checked with geonet
    def conv(self, x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.relu, normalizer_fn=None):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn, normalizer_fn=normalizer_fn)
    """

    def conv(self, x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.relu):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        # 对图像填充p
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn)

    def conv_block(self, x, num_out_layers, kernel_size):
        conv1 = self.conv(x, num_out_layers, kernel_size, 1)
        conv2 = self.conv(conv1, num_out_layers, kernel_size, 2)
        return conv2

    # it's checked with geonet
    def maxpool(self, x, kernel_size):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.max_pool2d(p_x, kernel_size)

    def convlayers_resblock(self, x, block_num, channels, stride):
        for i in range(block_num):
            x = slim.conv2d(x, channels, 2, stride, 'SAME', activation_fn=tf.nn.relu)
            skip1 = x
            x = slim.conv2d(x, channels, 2, stride, 'SAME', activation_fn=tf.nn.relu)
            skip2 = x
            x = slim.conv2d(x, channels, 2, stride, 'SAME', activation_fn=None)
            x = skip1 + skip2 + x
            x = tf.nn.relu(x)
        return x

    # it's checked with geonet
    def resconv(self, x, num_layers, stride):
        do_proj = tf.shape(x)[3] != num_layers or stride == 2
        shortcut = []
        conv1 = self.conv(x, num_layers, 1, 1)
        conv2 = self.conv(conv1, num_layers, 3, stride)
        conv3 = self.conv(conv2, 4 * num_layers, 1, 1, None)
        if do_proj:
            shortcut = self.conv(x, 4 * num_layers, 1, stride, None)
        else:
            shortcut = x
        return tf.nn.elu(conv3 + shortcut)

    # it's checked with geonet
    def resblock(self, x, num_layers, num_blocks):
        out = x
        for i in range(num_blocks - 1):
            out = self.resconv(out, num_layers, 1)
        out = self.resconv(out, num_layers, 2)
        return out

    # it's checked with geonet
    def upconv(self, x, num_out_layers, kernel_size, scale):
        upsample = self.upsample_nn(x, scale)
        conv = self.conv(upsample, num_out_layers, kernel_size, 1)
        return conv

    def up_conv_upsample(self, x, num_out_layers, kernel_size, scale):
        conv = self.conv(x, num_out_layers, kernel_size, 1)
        upsample = self.upsample_nn(conv, scale)
        return upsample

    def deconv(self, x, num_out_layers, kernel_size, scale):
        p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv = slim.conv2d_transpose(p_x, num_out_layers, kernel_size, scale, 'SAME')
        return conv[:, 3:-1, 3:-1, :]

    def build_resnet50(self):
        # set convenience functions
        conv = self.conv
        if self.params.use_deconv:
            upconv = self.deconv
        else:
            upconv = self.upconv

        with tf.variable_scope('preconv'):
            preconv = conv(self.input_nine, 64, 3, 1)
            conv1 = conv(preconv, 64, 7, 2)  # H/2  -   64D
            pool1 = self.maxpool(conv1, 3)  # H/4  -   64D
        with tf.variable_scope('encoder'):

            conv2 = self.resblock(pool1, 64, 3)  # H/8  -  256D
            conv3 = self.resblock(conv2, 128, 4)  # H/16 -  512D
            conv4 = self.resblock(conv3, 256, 6)  # H/32 - 1024D
            conv5 = self.resblock(conv4, 512, 3)  # H/64 - 2048D

        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = pool1
            skip3 = conv2
            skip4 = conv3
            skip5 = conv4

        # DECODING
        with tf.variable_scope('decoder1'):
            upconv6 = upconv(conv5, 512, 3, 2)  # H/32
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6 = conv(concat6, 512, 3, 1)
            disp6 = self.get_disp(iconv6)
            udisp6 = self.upsample_nn(disp6, 2)  # H/32

            upconv5 = upconv(iconv6, 256, 3, 2)  # H/16
            concat5 = tf.concat([upconv5, skip4, udisp6], 3)
            iconv5 = conv(concat5, 256, 3, 1)
            disp5 = self.get_disp(iconv5)
            udisp5 = self.upsample_nn(disp5, 2)  # H/32

            upconv4 = upconv(iconv5, 128, 3, 2)  # H/8
            concat4 = tf.concat([upconv4, skip3, udisp5], 3)
            iconv4 = conv(concat4, 128, 3, 1)

        with tf.variable_scope('disps_'):
            disp4 = self.get_disp(iconv4)
            udisp4 = self.upsample_nn(disp4, 2)
            upconv3 = upconv(iconv4, 64, 3, 2)  # H/4
            concat3 = tf.concat([upconv3, skip2, udisp4], 3)
            iconv3 = conv(concat3, 64, 3, 1)
            disp3 = self.get_disp(iconv3)
            udisp3 = self.upsample_nn(disp3, 2)

            upconv2 = upconv(iconv3, 48, 3, 2)  # H/2
            concat2 = tf.concat([upconv2, skip1, udisp3], 3)
            iconv2 = conv(concat2, 48, 3, 1)
            disp2 = self.get_disp(iconv2)

            udisp2 = self.upsample_nn(disp2, 2)

            upconv1 = upconv(iconv2, 32, 3, 2)  # H
            concat1 = tf.concat([upconv1, udisp2], 3)

            iconv1 = conv(concat1, 32, 3, 1)
            self.disp1 = self.get_disp(iconv1)
            # disp_net
            self.disp_center_est = tf.expand_dims(self.disp1[:, :, :, 0], 3)  # 0

            self.disp_1 = tf.expand_dims(self.disp1[:, :, :, 1], 3)  # 1
            self.disp_2 = tf.expand_dims(self.disp1[:, :, :, 2], 3)  # 2
            self.disp_3 = tf.expand_dims(self.disp1[:, :, :, 3], 3)  # 3
            self.disp_4 = tf.expand_dims(self.disp1[:, :, :, 4], 3)  # 4
            self.disp_5 = tf.expand_dims(self.disp1[:, :, :, 5], 3)  # 5
            self.disp_6 = tf.expand_dims(self.disp1[:, :, :, 6], 3)  # 6
            self.disp_7 = tf.expand_dims(self.disp1[:, :, :, 7], 3)  # 7
            self.disp_8 = tf.expand_dims(self.disp1[:, :, :, 8], 3)  # 8
            self.disp_9 = tf.expand_dims(self.disp1[:, :, :, 9], 3)  # 9
            self.disp_10 = tf.expand_dims(self.disp1[:, :, :, 10], 3)  # 10
            self.disp_11 = tf.expand_dims(self.disp1[:, :, :, 11], 3)  # 11
            self.disp_12 = tf.expand_dims(self.disp1[:, :, :, 12], 3)  # 12
            self.disp_13 = tf.expand_dims(self.disp1[:, :, :, 13], 3)  # 13
            self.disp_14 = tf.expand_dims(self.disp1[:, :, :, 14], 3)  # disp_center_est
            self.disp_15 = tf.expand_dims(self.disp1[:, :, :, 15], 3)  # 15
            self.disp_16 = tf.expand_dims(self.disp1[:, :, :, 16], 3)  # 16
            self.disp_17 = tf.expand_dims(self.disp1[:, :, :, 17], 3)  # 17
            self.disp_18 = tf.expand_dims(self.disp1[:, :, :, 18], 3)  # 18
            self.disp_19 = tf.expand_dims(self.disp1[:, :, :, 19], 3)  # 19
            self.disp_20 = tf.expand_dims(self.disp1[:, :, :, 20], 3)  # 20
            self.disp_21 = tf.expand_dims(self.disp1[:, :, :, 21], 3)  # 21
            self.disp_22 = tf.expand_dims(self.disp1[:, :, :, 22], 3)  # 22
            self.disp_23 = tf.expand_dims(self.disp1[:, :, :, 23], 3)  # 23
            self.disp_24 = tf.expand_dims(self.disp1[:, :, :, 24], 3)  # 24

            mask_disp1_flow = tf.concat([-3 * self.disp_1, -3 * self.disp_1], axis=3)
            mask_disp2_flow = tf.concat([-2 * self.disp_2, -2 * self.disp_2], axis=3)
            mask_disp3_flow = tf.concat([-1 * self.disp_3, -1 * self.disp_3], axis=3)
            mask_disp4_flow = tf.concat([1 * self.disp_4, 1 * self.disp_4], axis=3)
            mask_disp5_flow = tf.concat([2 * self.disp_5, 2 * self.disp_5], axis=3)
            mask_disp6_flow = tf.concat([3 * self.disp_6, 3 * self.disp_6], axis=3)
            mask_disp7_flow = tf.concat([3 * self.disp_7, -3 * self.disp_7], axis=3)
            mask_disp8_flow = tf.concat([2 * self.disp_8, -2 * self.disp_8], axis=3)
            mask_disp9_flow = tf.concat([1 * self.disp_9, -1 * self.disp_9], axis=3)
            mask_disp10_flow = tf.concat([-1 * self.disp_10, 1 * self.disp_10], axis=3)
            mask_disp11_flow = tf.concat([-2 * self.disp_11, 2 * self.disp_11], axis=3)
            mask_disp12_flow = tf.concat([-3 * self.disp_12, 3 * self.disp_12], axis=3)
            mask_disp13_flow = tf.concat([-3 * self.disp_13, tf.zeros_like(self.disp_13)], axis=3)
            mask_disp14_flow = tf.concat([-2 * self.disp_14, tf.zeros_like(self.disp_14)], axis=3)
            mask_disp15_flow = tf.concat([-1 * self.disp_15, tf.zeros_like(self.disp_15)], axis=3)
            mask_disp16_flow = tf.concat([1 * self.disp_16, tf.zeros_like(self.disp_16)], axis=3)
            mask_disp17_flow = tf.concat([2 * self.disp_17, tf.zeros_like(self.disp_17)], axis=3)
            mask_disp18_flow = tf.concat([3 * self.disp_18, tf.zeros_like(self.disp_18)], axis=3)
            mask_disp19_flow = tf.concat([tf.zeros_like(self.disp_19), -3 * self.disp_19], axis=3)
            mask_disp20_flow = tf.concat([tf.zeros_like(self.disp_20), -2 * self.disp_20], axis=3)
            mask_disp21_flow = tf.concat([tf.zeros_like(self.disp_21), -1 * self.disp_21], axis=3)
            mask_disp22_flow = tf.concat([tf.zeros_like(self.disp_22), 1 * self.disp_22], axis=3)
            mask_disp23_flow = tf.concat([tf.zeros_like(self.disp_23), 2 * self.disp_23], axis=3)
            mask_disp24_flow = tf.concat([tf.zeros_like(self.disp_24), 3 * self.disp_24], axis=3)
        with tf.variable_scope('rfine'):
            gray = tf.image.rgb_to_grayscale(self.model_input)

            # 利用灰度图和深度图stack在一起用三维卷积提取中心视差图
            stack_ref = tf.stack([tf.stop_gradient(self.disp1[:, :, :, 0:1]), gray], axis=1)
            stack_ref1 = slim.conv3d(stack_ref, 16, [3, 3, 3], 1)
            stack_ref2 = slim.conv3d(stack_ref1, 32, [3, 3, 3], 1)
            stack_ref3 = slim.conv3d(stack_ref2, 16, [3, 3, 3], 1)
            stack_ref4 = slim.conv3d(stack_ref3, 16, [2, 1, 1], 1, padding='VALID')

            # 对卷积后的tensor降维
            stack_ref4 = tf.squeeze(stack_ref4, axis=1)
            self.center_disp = self.get_centerdisp(stack_ref4)
            if self.mode == "train":
                # mask1
                # self.mask = tf.ones([25, 2, 384, 512, 1])
                self.mask = tf.stop_gradient(tf.py_func(func, [self.center_disp], tf.float32))

                # mask2 利用光流进行forward warping来得到mask
                self.mask_disp1 = tf.reshape(tf.clip_by_value(
                    transformerFwd(tf.ones(shape=[self.bz, 384, 512, 1], dtype='float32'),
                                   tf.stop_gradient(mask_disp1_flow),
                                   [384, 512]) * self.mask[1][:, :, :, 0:1], clip_value_min=0.0,
                    clip_value_max=1.0), [self.bz, 384, 512, 1])
                self.mask_disp2 = tf.reshape(tf.clip_by_value(
                    transformerFwd(tf.ones(shape=[self.bz, 384, 512, 1], dtype='float32'),
                                   tf.stop_gradient(mask_disp2_flow),
                                   [384, 512]) * self.mask[2][:, :, :, 0:1], clip_value_min=0.0,
                    clip_value_max=1.0), [self.bz, 384, 512, 1])
                self.mask_disp3 = tf.reshape(tf.clip_by_value(
                    transformerFwd(tf.ones(shape=[self.bz, 384, 512, 1], dtype='float32'),
                                   tf.stop_gradient(mask_disp3_flow),
                                   [384, 512]) * self.mask[3][:, :, :, 0:1], clip_value_min=0.0,
                    clip_value_max=1.0), [self.bz, 384, 512, 1])
                self.mask_disp4 = tf.reshape(tf.clip_by_value(
                    transformerFwd(tf.ones(shape=[self.bz, 384, 512, 1], dtype='float32'),
                                   tf.stop_gradient(mask_disp4_flow),
                                   [384, 512]) * self.mask[4][:, :, :, 0:1], clip_value_min=0.0,
                    clip_value_max=1.0), [self.bz, 384, 512, 1])
                self.mask_disp5 = tf.reshape(tf.clip_by_value(
                    transformerFwd(tf.ones(shape=[self.bz, 384, 512, 1], dtype='float32'),
                                   tf.stop_gradient(mask_disp5_flow),
                                   [384, 512]) * self.mask[5][:, :, :, 0:1], clip_value_min=0.0,
                    clip_value_max=1.0), [self.bz, 384, 512, 1])
                self.mask_disp6 = tf.reshape(tf.clip_by_value(
                    transformerFwd(tf.ones(shape=[self.bz, 384, 512, 1], dtype='float32'),
                                   tf.stop_gradient(mask_disp6_flow),
                                   [384, 512]) * self.mask[6][:, :, :, 0:1], clip_value_min=0.0,
                    clip_value_max=1.0), [self.bz, 384, 512, 1])
                self.mask_disp7 = tf.reshape(tf.clip_by_value(
                    transformerFwd(tf.ones(shape=[self.bz, 384, 512, 1], dtype='float32'),
                                   tf.stop_gradient(mask_disp7_flow),
                                   [384, 512]) * self.mask[7][:, :, :, 0:1], clip_value_min=0.0,
                    clip_value_max=1.0), [self.bz, 384, 512, 1])
                self.mask_disp8 = tf.reshape(tf.clip_by_value(
                    transformerFwd(tf.ones(shape=[self.bz, 384, 512, 1], dtype='float32'),
                                   tf.stop_gradient(mask_disp8_flow),
                                   [384, 512]) * self.mask[8][:, :, :, 0:1], clip_value_min=0.0,
                    clip_value_max=1.0), [self.bz, 384, 512, 1])
                self.mask_disp9 = tf.reshape(tf.clip_by_value(
                    transformerFwd(tf.ones(shape=[self.bz, 384, 512, 1], dtype='float32'),
                                   tf.stop_gradient(mask_disp9_flow),
                                   [384, 512]) * self.mask[9][:, :, :, 0:1], clip_value_min=0.0,
                    clip_value_max=1.0), [self.bz, 384, 512, 1])
                self.mask_disp10 = tf.reshape(tf.clip_by_value(
                    transformerFwd(tf.ones(shape=[self.bz, 384, 512, 1], dtype='float32'),
                                   tf.stop_gradient(mask_disp10_flow),
                                   [384, 512]) * self.mask[10][:, :, :, 0:1], clip_value_min=0.0,
                    clip_value_max=1.0), [self.bz, 384, 512, 1])
                self.mask_disp11 = tf.reshape(tf.clip_by_value(
                    transformerFwd(tf.ones(shape=[self.bz, 384, 512, 1], dtype='float32'),
                                   tf.stop_gradient(mask_disp11_flow),
                                   [384, 512]) * self.mask[11][:, :, :, 0:1], clip_value_min=0.0,
                    clip_value_max=1.0), [self.bz, 384, 512, 1])
                self.mask_disp12 = tf.reshape(tf.clip_by_value(
                    transformerFwd(tf.ones(shape=[self.bz, 384, 512, 1], dtype='float32'),
                                   tf.stop_gradient(mask_disp12_flow),
                                   [384, 512]) * self.mask[12][:, :, :, 0:1], clip_value_min=0.0,
                    clip_value_max=1.0), [self.bz, 384, 512, 1])
                self.mask_disp13 = tf.reshape(tf.clip_by_value(
                    transformerFwd(tf.ones(shape=[self.bz, 384, 512, 1], dtype='float32'),
                                   tf.stop_gradient(mask_disp13_flow),
                                   [384, 512]) * self.mask[13][:, :, :, 0:1], clip_value_min=0.0,
                    clip_value_max=1.0), [self.bz, 384, 512, 1])

                self.mask_disp13_visual = tf.reshape(tf.clip_by_value(
                    transformerFwd(tf.ones(shape=[self.bz, 384, 512, 1], dtype='float32'),
                                   tf.stop_gradient(mask_disp13_flow),
                                   [384, 512]), clip_value_min=0.0,
                    clip_value_max=1.0), [self.bz, 384, 512, 1])
                self.mask_disp14 = tf.reshape(tf.clip_by_value(
                    transformerFwd(tf.ones(shape=[self.bz, 384, 512, 1], dtype='float32'),
                                   tf.stop_gradient(mask_disp14_flow),
                                   [384, 512]) * self.mask[14][:, :, :, 0:1], clip_value_min=0.0,
                    clip_value_max=1.0), [self.bz, 384, 512, 1])
                self.mask_disp15 = tf.reshape(tf.clip_by_value(
                    transformerFwd(tf.ones(shape=[self.bz, 384, 512, 1], dtype='float32'),
                                   tf.stop_gradient(mask_disp15_flow),
                                   [384, 512]) * self.mask[15][:, :, :, 0:1], clip_value_min=0.0,
                    clip_value_max=1.0), [self.bz, 384, 512, 1])
                self.mask_disp16 = tf.reshape(tf.clip_by_value(
                    transformerFwd(tf.ones(shape=[self.bz, 384, 512, 1], dtype='float32'),
                                   tf.stop_gradient(mask_disp16_flow),
                                   [384, 512]) * self.mask[16][:, :, :, 0:1], clip_value_min=0.0,
                    clip_value_max=1.0), [self.bz, 384, 512, 1])
                self.mask_disp17 = tf.reshape(tf.clip_by_value(
                    transformerFwd(tf.ones(shape=[self.bz, 384, 512, 1], dtype='float32'),
                                   tf.stop_gradient(mask_disp17_flow),
                                   [384, 512]) * self.mask[17][:, :, :, 0:1], clip_value_min=0.0,
                    clip_value_max=1.0), [self.bz, 384, 512, 1])
                self.mask_disp18 = tf.reshape(tf.clip_by_value(
                    transformerFwd(tf.ones(shape=[self.bz, 384, 512, 1], dtype='float32'),
                                   tf.stop_gradient(mask_disp18_flow),
                                   [384, 512]) * self.mask[18][:, :, :, 0:1], clip_value_min=0.0,
                    clip_value_max=1.0), [self.bz, 384, 512, 1])

                self.mask_disp18_visual = tf.reshape(tf.clip_by_value(
                    transformerFwd(tf.ones(shape=[self.bz, 384, 512, 1], dtype='float32'),
                                   tf.stop_gradient(mask_disp18_flow),
                                   [384, 512]), clip_value_min=0.0,
                    clip_value_max=1.0), [self.bz, 384, 512, 1])
                self.mask_disp19 = tf.reshape(tf.clip_by_value(
                    transformerFwd(tf.ones(shape=[self.bz, 384, 512, 1], dtype='float32'),
                                   tf.stop_gradient(mask_disp19_flow),
                                   [384, 512]) * self.mask[19][:, :, :, 0:1], clip_value_min=0.0,
                    clip_value_max=1.0), [self.bz, 384, 512, 1])
                self.mask_disp20 = tf.reshape(tf.clip_by_value(
                    transformerFwd(tf.ones(shape=[self.bz, 384, 512, 1], dtype='float32'),
                                   tf.stop_gradient(mask_disp20_flow),
                                   [384, 512]) * self.mask[20][:, :, :, 0:1], clip_value_min=0.0,
                    clip_value_max=1.0), [self.bz, 384, 512, 1])
                self.mask_disp21 = tf.reshape(tf.clip_by_value(
                    transformerFwd(tf.ones(shape=[self.bz, 384, 512, 1], dtype='float32'),
                                   tf.stop_gradient(mask_disp21_flow),
                                   [384, 512]) * self.mask[21][:, :, :, 0:1], clip_value_min=0.0,
                    clip_value_max=1.0), [self.bz, 384, 512, 1])
                self.mask_disp22 = tf.reshape(tf.clip_by_value(
                    transformerFwd(tf.ones(shape=[self.bz, 384, 512, 1], dtype='float32'),
                                   tf.stop_gradient(mask_disp22_flow),
                                   [384, 512]) * self.mask[22][:, :, :, 0:1], clip_value_min=0.0,
                    clip_value_max=1.0), [self.bz, 384, 512, 1])
                self.mask_disp23 = tf.reshape(tf.clip_by_value(
                    transformerFwd(tf.ones(shape=[self.bz, 384, 512, 1], dtype='float32'),
                                   tf.stop_gradient(mask_disp23_flow),
                                   [384, 512]) * self.mask[23][:, :, :, 0:1], clip_value_min=0.0,
                    clip_value_max=1.0), [self.bz, 384, 512, 1])
                self.mask_disp24 = tf.reshape(tf.clip_by_value(
                    transformerFwd(tf.ones(shape=[self.bz, 384, 512, 1], dtype='float32'),
                                   tf.stop_gradient(mask_disp24_flow),
                                   [384, 512]) * self.mask[24][:, :, :, 0:1], clip_value_min=0.0,
                    clip_value_max=1.0), [self.bz, 384, 512, 1])

    def build_model(self):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):
            with tf.variable_scope('model', reuse=self.reuse_variables):
                self.model_input = self.center
                input_135 = [self.images_list[1], self.images_list[2], self.images_list[3],
                             self.images_list[4], self.images_list[5], self.images_list[6]
                             ]
                self.input_135 = tf.concat(input_135, axis=3)

                input_45 = [self.images_list[7], self.images_list[8], self.images_list[9],
                            self.images_list[10], self.images_list[11], self.images_list[12]
                            ]
                self.input_45 = tf.concat(input_45, axis=3)

                input_0 = [self.images_list[13], self.images_list[14], self.images_list[15],
                           self.images_list[16], self.images_list[17], self.images_list[18]
                           ]
                self.input_0 = tf.concat(input_0, axis=3)
                input_90 = [self.images_list[19], self.images_list[20], self.images_list[21],
                            self.images_list[22], self.images_list[23], self.images_list[24]
                            ]
                self.input_90 = tf.concat(input_90, axis=3)
                self.input_nine = tf.concat(
                    [self.model_input, self.input_135, self.input_45, self.input_0, self.input_90], axis=3)
                # build model
                self.build_resnet50()

    def build_outputs(self):
        # STORE DISPARITIES
        with tf.variable_scope('disparities'):
            # 9 out

            # center_disp是refine网络得到的中心视差图 disp_center_est是经过decoder得到的中心视差图
            self.two_centerdisp = tf.stack([self.center_disp, self.disp_center_est], axis=0)

            self.disp_est_list = [tf.expand_dims(self.disp1[:, :, :, i], 3) for i in range(1)]
            self.all_disp = tf.concat(
                [self.disp_1, self.disp_2, self.disp_3, self.disp_4, self.disp_5, self.disp_6,
                 self.disp_7, self.disp_8, self.disp_9, self.disp_10, self.disp_11, self.disp_12,
                 self.disp_13, self.disp_14, self.disp_15, self.disp_16, self.disp_17, self.disp_18,
                 self.disp_19, self.disp_20, self.disp_21, self.disp_22, self.disp_23, self.disp_24,
                 ], axis=3)

            # self.all_mask = tf.concat(
            #     [self.mask_disp1, self.mask_disp2, self.mask_disp3, self.mask_disp4, self.mask_disp5, self.mask_disp6,
            #      self.mask_disp7, self.mask_disp8, self.mask_disp9, self.mask_disp10, self.mask_disp11,
            #      self.mask_disp12,
            #      self.mask_disp13, self.mask_disp14, self.mask_disp15, self.mask_disp16, self.mask_disp17,
            #      self.mask_disp18,
            #      self.mask_disp19, self.mask_disp20, self.mask_disp21, self.mask_disp22, self.mask_disp23,
            #      self.mask_disp24,
            #      ], axis=3)

            """
            self.disp_topleft_est_2 = tf.expand_dims(self.disp2[:, :, :, 0], 3)
            self.disp_top_est_2 = tf.expand_dims(self.disp2[:, :, :, 1], 3)
            self.disp_topright_est_2 = tf.expand_dims(self.disp2[:, :, :, 2], 3)
            self.disp_left_est_2 = tf.expand_dims(self.disp2[:, :, :, 3], 3)
            self.disp_center_est_2 = tf.expand_dims(self.disp2[:, :, :, 4], 3)
            self.disp_right_est_2 = tf.expand_dims(self.disp2[:, :, :, 5], 3)
            self.disp_bottomleft_est_2 = tf.expand_dims(self.disp2[:, :, :, 6], 3)
            self.disp_bottom_est_2 = tf.expand_dims(self.disp2[:, :, :, 7], 3)
            self.disp_bottomright_est_2 = tf.expand_dims(self.disp2[:, :, :, 8], 3)

            self.disp_est_list_2 = [tf.expand_dims(self.disp2[:, :, :, i], 3) for i in range(9)]

            """

    ######################
    # COMPUTE IMAGE LOSS #
    ######################
    # compute image loss of each pixel: l1 + ssim
    def comput_imageloss(self, im, im_est):
        # GENERATE L1 DIFFERENCE
        l1_im = tf.abs(im_est - im)
        # GENERATE SSIM
        ssim_im = self.SSIM_no_mask(im_est, im)
        # IMAGE LOSS OF EACH PIXEL: WEIGTHED SUM
        return self.params.alpha_image_loss * ssim_im + (1 - self.params.alpha_image_loss) * l1_im

    def comput_imageloss_mask_refine(self, im, im_est, mask):
        # GENERATE L1 DIFFERENCE
        # l1_im = tf.abs(im_est - im) * mask
        # GENERATE SSIM
        ssim_im = self.SSIM_no_mask(im_est * mask, im * mask)
        # IMAGE LOSS OF EACH PIXEL: WEIGTHED SUM
        return self.params.alpha_image_loss * ssim_im

    def comput_imageloss_mask(self, im, im_est, mask):
        # GENERATE L1 DIFFERENCE
        l1_im = tf.abs(im_est - im) * mask
        # GENERATE SSIM
        ssim_im = self.SSIM_no_mask(im_est * mask, im * mask)
        # IMAGE LOSS OF EACH PIXEL: WEIGTHED SUM
        return self.params.alpha_image_loss * ssim_im + (1 - self.params.alpha_image_loss) * l1_im

    # compute image loss: l1 + ssim

    # def comput_l2_loss(self, im, im_est, mask=None):
    #     if mask == None:
    #         image_pixel = tf.square(tf.abs(im - im_est) + 0.0000001)
    #         return tf.reduce_mean(image_pixel[:, 10:-10, 10:-10, :] + 0.0000001)
    #     else:
    #         image_pixel = tf.square(tf.abs(im - im_est) + 0.0000001) * mask
    #         return tf.reduce_mean(image_pixel[:, 10:-10, 10:-10, :]) / tf.reduce_mean(mask[:, 10:-10, 10:-10, :])

    def comput_l2_loss(self, im, im_est, mask=None):
        if mask == None:
            image_pixel = tf.square(tf.abs(im - im_est) + 0.0000001)
            return image_pixel
        else:
            image_pixel = (tf.square(tf.abs(im - im_est) + 0.0000001) * mask) / self.mask_sum
            return image_pixel

    def comput_imageloss_mean(self, im, im_est):
        # IMAGE LOSS OF EACH PIXEL: WEIGTHED SUM
        image_loss_pixel = self.comput_imageloss(im, im_est)
        # MEAN
        return tf.reduce_mean(image_loss_pixel[:, 10:-10, 10:-10, :])
        # return tf.reduce_mean(image_loss_pixel)

    def comput_imageloss_2(self, im, im_est):
        # GENERATE L1 DIFFERENCE
        l1_im = tf.abs(im_est - im)
        # GENERATE SSIM
        # ssim_im = self.SSIM(im_est, im)
        # IMAGE LOSS OF EACH PIXEL: WEIGTHED SUM
        return l1_im

    def comput_imageloss_mean_mask_refine(self, im, im_est, mask):
        # IMAGE LOSS OF EACH PIXEL: WEIGTHED SUM
        image_loss_pixel = self.comput_imageloss_mask_refine(im, im_est, mask)
        return tf.reduce_mean(image_loss_pixel[:, 10:-10, 10:-10, :]) / (
                tf.reduce_mean(mask[:, 10:-10, 10:-10, :]) + 1e-6)

    def comput_imageloss_mean_mask(self, im, im_est, mask):
        # IMAGE LOSS OF EACH PIXEL: WEIGTHED SUM
        image_loss_pixel = self.comput_imageloss_mask(im, im_est, mask)
        return tf.reduce_mean(image_loss_pixel[:, 10:-10, 10:-10, :]) / (
                tf.reduce_mean(mask[:, 10:-10, 10:-10, :]) + 1e-6)

    def comput_CAD(self, im, im_est):
        # GENERATE L1 DIFFERENCE
        l1_im = tf.abs(im_est - im)  # RGB different
        l1_im = -l1_im
        zu_of_num = 3

        temp = tf.expand_dims(tf.reduce_mean(l1_im[:, 0 * zu_of_num:(0 + 1) * zu_of_num, :, :, :], axis=1), axis=1)
        # temp /= 3.0  # each direcention mean
        color_constraint = tf.expand_dims(tf.reduce_mean(l1_im[:, 0 * zu_of_num:(0 + 1) * zu_of_num, :, :, :], axis=4),
                                          axis=4)
        with_constraint = temp + 0.1 * tf.reduce_mean(color_constraint)

        for i in range(1, int(int(l1_im.shape[1]) / zu_of_num)):
            temp = tf.expand_dims(tf.reduce_mean(l1_im[:, i * zu_of_num:(i + 1) * zu_of_num, :, :, :], axis=1), axis=1)
            # temp /= 3.0
            color_constraint = tf.expand_dims(
                tf.reduce_mean(l1_im[:, i * zu_of_num:(i + 1) * zu_of_num, :, :, :], axis=4), axis=4)
            temp = temp + 0.1 * tf.reduce_mean(color_constraint)
            with_constraint = tf.concat([with_constraint, temp], axis=1)

        adaptive_refocusloss = slim.max_pool3d(with_constraint, kernel_size=[8, 1, 1],
                                               stride=[1, 1, 1])  # the min direction mean
        adaptive_refocusloss = -adaptive_refocusloss * 8  # (4, 244, 244 ,3)

        return tf.reduce_mean(adaptive_refocusloss)

    def comput_CAE(self, im, im_est):
        # GENERATE L1 DIFFERENCE
        l1_im = tf.abs(im_est - im)  # RGB different (4, 24, 244, 244, 3)

        # variance = tf.sqrt(tf.reduce_sum(tf.square(l1_im), axis=1) / 24.0) * 3.0     # (4, 244, 244, 3)
        # variance = tf.expand_dims(variance, axis=1)

        # mean, variance = tf.nn.moments(l1_im, axes=1)
        # variance = tf.expand_dims(variance, axis=1) * 3.0
        # fine = tf.maximum(0.0, 1.0 - tf.divide(l1_im, variance + 1e-5))

        variance = 0.5
        fine = tf.maximum(0.0, 1.0 - tf.divide(l1_im, variance))
        h1 = tf.divide(tf.reduce_sum(fine, axis=1), 24.0)
        # h1 = tf.divide(tf.reduce_sum(fine, axis=1), 24.0)
        # h0 = h1 * 0.6 + 0.4

        wi = tf.exp(-tf.divide(tf.square(l1_im), 2 * variance))
        wi = tf.divide(tf.reduce_sum(wi, axis=1), 24.0)

        gi = wi * h1
        # gi = tf.divide(gi, 2.0)
        # gi = tf.reduce_sum(gi, axis=1)
        gi = gi * 0.6 + 0.4

        AE_loss = - gi * tf.log(gi)
        CAE_loss = tf.reduce_mean(AE_loss)

        # AE_loss = - h0 * tf.log(h0)
        # CAE_loss = tf.reduce_mean(AE_loss)

        l1_pie = (1. - fine) * l1_im
        l1_pie_loss = tf.reduce_mean(l1_pie)

        return CAE_loss + l1_pie_loss

    def comput_CAD_loss(self, centerdisp, refocus):
        # IMAGE LOSS OF EACH PIXEL: WEIGTHED SUM
        centerdisp = centerdisp[:, 12:500, 12:500, :]
        centerdisp = tf.expand_dims(centerdisp, axis=1)
        image_loss_pixel = self.comput_CAD(centerdisp, refocus[:, :, 12:500, 12:500, :])
        ######### occu loss
        # MEAN
        # return tf.reduce_mean(image_loss_pixel[:,10:-10,10:-10,:])
        return image_loss_pixel

    def comput_CAE_loss(self, centerdisp, refocus):
        # IMAGE LOSS OF EACH PIXEL: WEIGTHED SUM
        centerdisp = centerdisp[:, 12:500, 12:500, :]
        centerdisp = tf.expand_dims(centerdisp, axis=1)
        CAE_L1_pie = self.comput_CAE(centerdisp, refocus[:, :, 12:500, 12:500, :])
        ######### occu loss
        # MEAN
        # return tf.reduce_mean(image_loss_pixel[:,10:-10,10:-10,:])
        return CAE_L1_pie

    ######################################
    # COMPUTE DISPARITY CONSISTENCE LOSS #
    ######################################
    # left-right consistence check
    def comput_dp_consistence(self, disp, disp_est):
        return tf.abs(disp - disp_est)

    def nine_output_disp(self, center_im):
        pred = []
        center_im_disp = tf.concat([center_im, tf.stop_gradient(self.disp_center_est)], axis=3)
        image_list_disp = []
        image_list_disp.append(tf.concat([self.images_list[1], tf.stop_gradient(self.disp_1)], axis=3))
        image_list_disp.append(tf.concat([self.images_list[2], tf.stop_gradient(self.disp_2)], axis=3))
        image_list_disp.append(tf.concat([self.images_list[3], tf.stop_gradient(self.disp_3)], axis=3))
        image_list_disp.append(tf.concat([self.images_list[4], tf.stop_gradient(self.disp_4)], axis=3))
        image_list_disp.append(tf.concat([self.images_list[5], tf.stop_gradient(self.disp_5)], axis=3))
        image_list_disp.append(tf.concat([self.images_list[6], tf.stop_gradient(self.disp_6)], axis=3))
        image_list_disp.append(tf.concat([self.images_list[7], tf.stop_gradient(self.disp_7)], axis=3))
        image_list_disp.append(tf.concat([self.images_list[8], tf.stop_gradient(self.disp_8)], axis=3))
        image_list_disp.append(tf.concat([self.images_list[9], tf.stop_gradient(self.disp_9)], axis=3))
        image_list_disp.append(tf.concat([self.images_list[10], tf.stop_gradient(self.disp_10)], axis=3))
        image_list_disp.append(tf.concat([self.images_list[11], tf.stop_gradient(self.disp_11)], axis=3))
        image_list_disp.append(tf.concat([self.images_list[12], tf.stop_gradient(self.disp_12)], axis=3))
        image_list_disp.append(tf.concat([self.images_list[13], tf.stop_gradient(self.disp_13)], axis=3))
        image_list_disp.append(tf.concat([self.images_list[14], tf.stop_gradient(self.disp_14)], axis=3))
        image_list_disp.append(tf.concat([self.images_list[15], tf.stop_gradient(self.disp_15)], axis=3))
        image_list_disp.append(tf.concat([self.images_list[16], tf.stop_gradient(self.disp_16)], axis=3))
        image_list_disp.append(tf.concat([self.images_list[17], tf.stop_gradient(self.disp_17)], axis=3))
        image_list_disp.append(tf.concat([self.images_list[18], tf.stop_gradient(self.disp_18)], axis=3))
        image_list_disp.append(tf.concat([self.images_list[19], tf.stop_gradient(self.disp_19)], axis=3))
        image_list_disp.append(tf.concat([self.images_list[20], tf.stop_gradient(self.disp_20)], axis=3))
        image_list_disp.append(tf.concat([self.images_list[21], tf.stop_gradient(self.disp_21)], axis=3))
        image_list_disp.append(tf.concat([self.images_list[22], tf.stop_gradient(self.disp_22)], axis=3))
        image_list_disp.append(tf.concat([self.images_list[23], tf.stop_gradient(self.disp_23)], axis=3))
        image_list_disp.append(tf.concat([self.images_list[24], tf.stop_gradient(self.disp_24)], axis=3))

        pred.append(self.generate_image_topleft(center_im_disp, self.disp_1 * 3, self.disp_1 * 3))
        pred.append(self.generate_image_topleft(center_im_disp, self.disp_2 * 2, self.disp_2 * 2))
        pred.append(self.generate_image_topleft(center_im_disp, self.disp_3 * 1, self.disp_3 * 1))
        pred.append(self.generate_image_bottomright(center_im_disp, self.disp_4 * 1, self.disp_4 * 1))
        pred.append(self.generate_image_bottomright(center_im_disp, self.disp_5 * 2, self.disp_5 * 2))
        pred.append(self.generate_image_bottomright(center_im_disp, self.disp_6 * 3, self.disp_6 * 3))
        pred.append(self.generate_image_topright(center_im_disp, self.disp_7 * 3, self.disp_7 * 3))
        pred.append(self.generate_image_topright(center_im_disp, self.disp_8 * 2, self.disp_8 * 2))
        pred.append(self.generate_image_topright(center_im_disp, self.disp_9 * 1, self.disp_9 * 1))
        pred.append(self.generate_image_bottomleft(center_im_disp, self.disp_10 * 1, self.disp_10 * 1))
        pred.append(self.generate_image_bottomleft(center_im_disp, self.disp_11 * 2, self.disp_11 * 2))
        pred.append(self.generate_image_bottomleft(center_im_disp, self.disp_12 * 3, self.disp_12 * 3))
        pred.append(self.generate_image_left(center_im_disp, self.disp_13 * 3))
        pred.append(self.generate_image_left(center_im_disp, self.disp_14 * 2))
        pred.append(self.generate_image_left(center_im_disp, self.disp_15 * 1))
        pred.append(self.generate_image_right(center_im_disp, self.disp_16 * 1))
        pred.append(self.generate_image_right(center_im_disp, self.disp_17 * 2))
        pred.append(self.generate_image_right(center_im_disp, self.disp_18 * 3))
        pred.append(self.generate_image_top(center_im_disp, self.disp_19 * 3))
        pred.append(self.generate_image_top(center_im_disp, self.disp_20 * 2))
        pred.append(self.generate_image_top(center_im_disp, self.disp_21 * 1))
        pred.append(self.generate_image_bottom(center_im_disp, self.disp_22 * 1))
        pred.append(self.generate_image_bottom(center_im_disp, self.disp_23 * 2))
        pred.append(self.generate_image_bottom(center_im_disp, self.disp_24 * 3))
        loss = 0
        for i in range(1, 25):
            loss += self.comput_imageloss_mean(pred[i - 1], image_list_disp[i - 1])
        return loss

    def left_right_consistent(self):
        pred_disp_center = [self.disp_center_est]
        pred_disp_center.append(
            self.generate_image_bottomright(tf.stop_gradient(self.disp_1), self.disp_center_est * 3,
                                            self.disp_center_est * 3))
        pred_disp_center.append(
            self.generate_image_bottomright(self.disp_2, self.disp_center_est * 2, self.disp_center_est * 2))
        pred_disp_center.append(
            self.generate_image_bottomright(self.disp_3, self.disp_center_est * 1, self.disp_center_est * 1))
        pred_disp_center.append(
            self.generate_image_topleft(self.disp_4, self.disp_center_est * 1, self.disp_center_est * 1))
        pred_disp_center.append(
            self.generate_image_topleft(self.disp_5, self.disp_center_est * 2, self.disp_center_est * 2))
        pred_disp_center.append(
            self.generate_image_topleft(self.disp_6, self.disp_center_est * 3, self.disp_center_est * 3))
        pred_disp_center.append(
            self.generate_image_bottomleft(self.disp_7, self.disp_center_est * 3, self.disp_center_est * 3))
        pred_disp_center.append(
            self.generate_image_bottomleft(self.disp_8, self.disp_center_est * 2, self.disp_center_est * 2))
        pred_disp_center.append(
            self.generate_image_bottomleft(self.disp_9, self.disp_center_est * 1, self.disp_center_est * 1))
        pred_disp_center.append(
            self.generate_image_topright(self.disp_10, self.disp_center_est * 1, self.disp_center_est * 1))
        pred_disp_center.append(
            self.generate_image_topright(self.disp_11, self.disp_center_est * 2, self.disp_center_est * 2))
        pred_disp_center.append(
            self.generate_image_topright(self.disp_12, self.disp_center_est * 3, self.disp_center_est * 3))
        pred_disp_center.append(self.generate_image_right(self.disp_13, self.disp_center_est * 3))
        pred_disp_center.append(self.generate_image_right(self.disp_14, self.disp_center_est * 2))
        pred_disp_center.append(self.generate_image_right(self.disp_15, self.disp_center_est * 1))
        pred_disp_center.append(self.generate_image_left(self.disp_16, self.disp_center_est * 1))
        pred_disp_center.append(self.generate_image_left(self.disp_17, self.disp_center_est * 2))
        pred_disp_center.append(self.generate_image_left(self.disp_18, self.disp_center_est * 3))
        pred_disp_center.append(self.generate_image_bottom(self.disp_19, self.disp_center_est * 3))
        pred_disp_center.append(self.generate_image_bottom(self.disp_20, self.disp_center_est * 2))
        pred_disp_center.append(self.generate_image_bottom(self.disp_21, self.disp_center_est * 1))
        pred_disp_center.append(self.generate_image_top(self.disp_22, self.disp_center_est * 1))
        pred_disp_center.append(self.generate_image_top(self.disp_23, self.disp_center_est * 2))
        pred_disp_center.append(self.generate_image_top(self.disp_24, self.disp_center_est * 3))
        mean_disp_center = pred_disp_center[0]
        mean_disp_center += pred_disp_center[1] * self.mask_disp1
        mean_disp_center += pred_disp_center[2] * self.mask_disp2
        mean_disp_center += pred_disp_center[3] * self.mask_disp3
        mean_disp_center += pred_disp_center[4] * self.mask_disp4
        mean_disp_center += pred_disp_center[5] * self.mask_disp5
        mean_disp_center += pred_disp_center[6] * self.mask_disp6
        mean_disp_center += pred_disp_center[7] * self.mask_disp7
        mean_disp_center += pred_disp_center[8] * self.mask_disp8
        mean_disp_center += pred_disp_center[9] * self.mask_disp9
        mean_disp_center += pred_disp_center[10] * self.mask_disp10
        mean_disp_center += pred_disp_center[11] * self.mask_disp11
        mean_disp_center += pred_disp_center[12] * self.mask_disp12
        mean_disp_center += pred_disp_center[13] * self.mask_disp13
        mean_disp_center += pred_disp_center[14] * self.mask_disp14
        mean_disp_center += pred_disp_center[15] * self.mask_disp15
        mean_disp_center += pred_disp_center[16] * self.mask_disp16
        mean_disp_center += pred_disp_center[17] * self.mask_disp17
        mean_disp_center += pred_disp_center[18] * self.mask_disp18
        mean_disp_center += pred_disp_center[19] * self.mask_disp19
        mean_disp_center += pred_disp_center[20] * self.mask_disp20
        mean_disp_center += pred_disp_center[21] * self.mask_disp21
        mean_disp_center += pred_disp_center[22] * self.mask_disp22
        mean_disp_center += pred_disp_center[23] * self.mask_disp23
        mean_disp_center += pred_disp_center[24] * self.mask_disp24

        self.mean_disp_center = mean_disp_center / self.mask_sum
        self.consistent_loss = tf.zeros_like(mean_disp_center, dtype=tf.float32)
        self.consistent_loss += self.comput_l2_loss(pred_disp_center[0], self.mean_disp_center)
        self.consistent_loss += self.comput_l2_loss(pred_disp_center[1], self.mean_disp_center,
                                                    self.mask_disp1)
        self.consistent_loss += self.comput_l2_loss(pred_disp_center[2], self.mean_disp_center,
                                                    self.mask_disp2)
        self.consistent_loss += self.comput_l2_loss(pred_disp_center[3], self.mean_disp_center,
                                                    self.mask_disp3)
        self.consistent_loss += self.comput_l2_loss(pred_disp_center[4], self.mean_disp_center,
                                                    self.mask_disp4)
        self.consistent_loss += self.comput_l2_loss(pred_disp_center[5], self.mean_disp_center,
                                                    self.mask_disp5)
        self.consistent_loss += self.comput_l2_loss(pred_disp_center[6], self.mean_disp_center,
                                                    self.mask_disp6)
        self.consistent_loss += self.comput_l2_loss(pred_disp_center[7], self.mean_disp_center,
                                                    self.mask_disp7)
        self.consistent_loss += self.comput_l2_loss(pred_disp_center[8], self.mean_disp_center,
                                                    self.mask_disp8)
        self.consistent_loss += self.comput_l2_loss(pred_disp_center[9], self.mean_disp_center,
                                                    self.mask_disp9)
        self.consistent_loss += self.comput_l2_loss(pred_disp_center[10], self.mean_disp_center,
                                                    self.mask_disp10)
        self.consistent_loss += self.comput_l2_loss(pred_disp_center[11], self.mean_disp_center,
                                                    self.mask_disp11)
        self.consistent_loss += self.comput_l2_loss(pred_disp_center[12], self.mean_disp_center,
                                                    self.mask_disp12)
        self.consistent_loss += self.comput_l2_loss(pred_disp_center[13], self.mean_disp_center,
                                                    self.mask_disp13)
        self.consistent_loss += self.comput_l2_loss(pred_disp_center[14], self.mean_disp_center,
                                                    self.mask_disp14)
        self.consistent_loss += self.comput_l2_loss(pred_disp_center[15], self.mean_disp_center,
                                                    self.mask_disp15)
        self.consistent_loss += self.comput_l2_loss(pred_disp_center[16], self.mean_disp_center,
                                                    self.mask_disp16)
        self.consistent_loss += self.comput_l2_loss(pred_disp_center[17], self.mean_disp_center,
                                                    self.mask_disp17)
        self.consistent_loss += self.comput_l2_loss(pred_disp_center[18], self.mean_disp_center,
                                                    self.mask_disp18)
        self.consistent_loss += self.comput_l2_loss(pred_disp_center[19], self.mean_disp_center,
                                                    self.mask_disp19)
        self.consistent_loss += self.comput_l2_loss(pred_disp_center[20], self.mean_disp_center,
                                                    self.mask_disp20)
        self.consistent_loss += self.comput_l2_loss(pred_disp_center[21], self.mean_disp_center,
                                                    self.mask_disp21)
        self.consistent_loss += self.comput_l2_loss(pred_disp_center[22], self.mean_disp_center,
                                                    self.mask_disp22)
        self.consistent_loss += self.comput_l2_loss(pred_disp_center[23], self.mean_disp_center,
                                                    self.mask_disp23)
        self.consistent_loss += self.comput_l2_loss(pred_disp_center[24], self.mean_disp_center,
                                                    self.mask_disp24)

        return tf.reduce_mean(self.consistent_loss[:, 10:-10, 10:-10, :])

    def nine_output(self, center_im):
        pred = []
        pred.append(self.generate_image_topleft(center_im, self.disp_1 * 3, self.disp_1 * 3))
        pred.append(self.generate_image_topleft(center_im, self.disp_2 * 2, self.disp_2 * 2))
        pred.append(self.generate_image_topleft(center_im, self.disp_3 * 1, self.disp_3 * 1))
        pred.append(self.generate_image_bottomright(center_im, self.disp_4 * 1, self.disp_4 * 1))
        pred.append(self.generate_image_bottomright(center_im, self.disp_5 * 2, self.disp_5 * 2))
        pred.append(self.generate_image_bottomright(center_im, self.disp_6 * 3, self.disp_6 * 3))
        pred.append(self.generate_image_topright(center_im, self.disp_7 * 3, self.disp_7 * 3))
        pred.append(self.generate_image_topright(center_im, self.disp_8 * 2, self.disp_8 * 2))
        pred.append(self.generate_image_topright(center_im, self.disp_9 * 1, self.disp_9 * 1))
        pred.append(self.generate_image_bottomleft(center_im, self.disp_10 * 1, self.disp_10 * 1))
        pred.append(self.generate_image_bottomleft(center_im, self.disp_11 * 2, self.disp_11 * 2))
        pred.append(self.generate_image_bottomleft(center_im, self.disp_12 * 3, self.disp_12 * 3))
        pred.append(self.generate_image_left(center_im, self.disp_13 * 3))
        pred.append(self.generate_image_left(center_im, self.disp_14 * 2))
        pred.append(self.generate_image_left(center_im, self.disp_15 * 1))
        pred.append(self.generate_image_right(center_im, self.disp_16 * 1))
        pred.append(self.generate_image_right(center_im, self.disp_17 * 2))
        pred.append(self.generate_image_right(center_im, self.disp_18 * 3))
        pred.append(self.generate_image_top(center_im, self.disp_19 * 3))
        pred.append(self.generate_image_top(center_im, self.disp_20 * 2))
        pred.append(self.generate_image_top(center_im, self.disp_21 * 1))
        pred.append(self.generate_image_bottom(center_im, self.disp_22 * 1))
        pred.append(self.generate_image_bottom(center_im, self.disp_23 * 2))
        pred.append(self.generate_image_bottom(center_im, self.disp_24 * 3))
        loss = 0
        for i in range(1, 25):
            loss += self.comput_imageloss_mean(pred[i - 1], self.images_list[i])
        return loss

    def get_24_smoothlos(self):
        loss = 0
        loss += self.get_disparity_smoothness(self.disp_center_est, self.images_list[0])
        loss += self.get_disparity_smoothness(self.disp_1, self.images_list[1])
        loss += self.get_disparity_smoothness(self.disp_2, self.images_list[2])
        loss += self.get_disparity_smoothness(self.disp_3, self.images_list[3])
        loss += self.get_disparity_smoothness(self.disp_4, self.images_list[4])
        loss += self.get_disparity_smoothness(self.disp_5, self.images_list[5])
        loss += self.get_disparity_smoothness(self.disp_6, self.images_list[6])
        loss += self.get_disparity_smoothness(self.disp_7, self.images_list[7])
        loss += self.get_disparity_smoothness(self.disp_8, self.images_list[8])
        loss += self.get_disparity_smoothness(self.disp_9, self.images_list[9])
        loss += self.get_disparity_smoothness(self.disp_10, self.images_list[10])
        loss += self.get_disparity_smoothness(self.disp_11, self.images_list[11])
        loss += self.get_disparity_smoothness(self.disp_12, self.images_list[12])
        loss += self.get_disparity_smoothness(self.disp_13, self.images_list[13])
        loss += self.get_disparity_smoothness(self.disp_14, self.images_list[14])
        loss += self.get_disparity_smoothness(self.disp_15, self.images_list[15])
        loss += self.get_disparity_smoothness(self.disp_16, self.images_list[16])
        loss += self.get_disparity_smoothness(self.disp_17, self.images_list[17])
        loss += self.get_disparity_smoothness(self.disp_18, self.images_list[18])
        loss += self.get_disparity_smoothness(self.disp_19, self.images_list[19])
        loss += self.get_disparity_smoothness(self.disp_20, self.images_list[20])
        loss += self.get_disparity_smoothness(self.disp_21, self.images_list[21])
        loss += self.get_disparity_smoothness(self.disp_22, self.images_list[22])
        loss += self.get_disparity_smoothness(self.disp_23, self.images_list[23])
        loss += self.get_disparity_smoothness(self.disp_24, self.images_list[24])
        return loss

    def comput_loss_45degree_orientation(self, center_im, center_disp):
        # get images
        topleft_im_3 = self.images_list[1]  # 000
        topleft_im_2 = self.images_list[2]  # 010
        topleft_im_1 = self.images_list[3]  # 020
        bottomright_im_1 = self.images_list[4]  # 040
        bottomright_im_2 = self.images_list[5]  # 050
        bottomright_im_3 = self.images_list[6]  # 060

        # GENERATE ESTIMATED CENTER IMAGES
        centerimg_est_from_topleft_1 = self.generate_image_bottomright(topleft_im_1, center_disp, center_disp)
        centerimg_est_from_topleft_2 = self.generate_image_bottomright(topleft_im_2, center_disp * 2., center_disp * 2.)
        centerimg_est_from_topleft_3 = self.generate_image_bottomright(topleft_im_3, center_disp * 3., center_disp * 3.)
        centerimg_est_from_bottomright_1 = self.generate_image_topleft(bottomright_im_1, center_disp, center_disp)
        centerimg_est_from_bottomright_2 = self.generate_image_topleft(bottomright_im_2, center_disp * 2.,
                                                                       center_disp * 2.)
        centerimg_est_from_bottomright_3 = self.generate_image_topleft(bottomright_im_3, center_disp * 3.,
                                                                       center_disp * 3.)
        # COMPUTE IMAGE LOSS
        imloss_center_with_topleft_1 = self.comput_imageloss_mean_mask(center_im,
                                                                       centerimg_est_from_topleft_1,
                                                                       self.mask_disp3)  # if disocc not need
        imloss_center_with_topleft_2 = self.comput_imageloss_mean_mask(center_im, centerimg_est_from_topleft_2,
                                                                       self.mask_disp2)
        imloss_center_with_topleft_3 = self.comput_imageloss_mean_mask(center_im, centerimg_est_from_topleft_3,
                                                                       self.mask_disp1)
        imloss_center_with_bottomright_1 = self.comput_imageloss_mean_mask(center_im,
                                                                           centerimg_est_from_bottomright_1,
                                                                           self.mask_disp4)  # if disocc not need
        imloss_center_with_bottomright_2 = self.comput_imageloss_mean_mask(center_im, centerimg_est_from_bottomright_2,
                                                                           self.mask_disp5)
        imloss_center_with_bottomright_3 = self.comput_imageloss_mean_mask(center_im, centerimg_est_from_bottomright_3,
                                                                           self.mask_disp6)

        self.refocusedim_45 = (
                centerimg_est_from_topleft_1 + centerimg_est_from_topleft_2 + centerimg_est_from_topleft_3 +
                centerimg_est_from_bottomright_1 + centerimg_est_from_bottomright_2 + centerimg_est_from_bottomright_3)

        return (imloss_center_with_topleft_1 + imloss_center_with_topleft_2 * 2.0 + imloss_center_with_topleft_3 * 3.0 +
                imloss_center_with_bottomright_1 + imloss_center_with_bottomright_2 * 2.0 + imloss_center_with_bottomright_3 * 3.0)

    def comput_loss_135degree_orientation(self, center_im, center_disp):
        # get images
        topright_im_3 = self.images_list[7]  # 006
        topright_im_2 = self.images_list[8]  # 014
        topright_im_1 = self.images_list[9]  # 022
        bottomleft_im_1 = self.images_list[10]  # 038
        bottomleft_im_2 = self.images_list[11]  # 046
        bottomleft_im_3 = self.images_list[12]  # 054
        # GENERATE ESTIMATED CENTER IMAGES
        centerimg_est_from_topright_1 = self.generate_image_bottomleft(topright_im_1, center_disp, center_disp)
        centerimg_est_from_topright_2 = self.generate_image_bottomleft(topright_im_2, center_disp * 2.,
                                                                       center_disp * 2.)
        centerimg_est_from_topright_3 = self.generate_image_bottomleft(topright_im_3, center_disp * 3.,
                                                                       center_disp * 3.)
        centerimg_est_from_bottomleft_1 = self.generate_image_topright(bottomleft_im_1, center_disp, center_disp)
        centerimg_est_from_bottomleft_2 = self.generate_image_topright(bottomleft_im_2, center_disp * 2.,
                                                                       center_disp * 2.)
        centerimg_est_from_bottomleft_3 = self.generate_image_topright(bottomleft_im_3, center_disp * 3.,
                                                                       center_disp * 3.)

        # GENERATE ESTIMATED TOPRIGHT AND BOTTOMLEFT IMAGES
        # imageListNumber_7_est_form_center = self.generate_image_topright(center_im, self.disp_7_est * 3.,
        #                                                                  self.disp_7_est * 3.)
        # imageListNumber_8_est_form_center = self.generate_image_topright(center_im, self.disp_8_est * 2.,
        #                                                                  self.disp_8_est * 2.)
        # toprightimg_1_est_from_center = self.generate_image_topright(center_im, topright_disp, topright_disp)
        #
        # bottomleftimg_1_est_from_center = self.generate_image_bottomleft(center_im, bottomleft_disp, bottomleft_disp)
        # imageListNumber_11_est_form_center = self.generate_image_bottomleft(center_im, self.disp_11_est * 2.,
        #                                                                     self.disp_11_est * 2.)
        # imageListNumber_12_est_form_center = self.generate_image_bottomleft(center_im, self.disp_12_est * 3.,
        #                                                                     self.disp_12_est * 3.)

        # COMPUTE IMAGE LOSS （包括L1-distense SSIM）
        imloss_center_with_topright_1 = self.comput_imageloss_mean_mask(center_im,
                                                                        centerimg_est_from_topright_1,
                                                                        self.mask_disp9)  # if disocc not need
        imloss_center_with_topright_2 = self.comput_imageloss_mean_mask(center_im, centerimg_est_from_topright_2,
                                                                        self.mask_disp8)
        imloss_center_with_topright_3 = self.comput_imageloss_mean_mask(center_im, centerimg_est_from_topright_3,
                                                                        self.mask_disp7)
        imloss_center_with_bottomleft_1 = self.comput_imageloss_mean_mask(center_im,
                                                                          centerimg_est_from_bottomleft_1,
                                                                          self.mask_disp10)  # if disocc not need
        imloss_center_with_bottomleft_2 = self.comput_imageloss_mean_mask(center_im, centerimg_est_from_bottomleft_2,
                                                                          self.mask_disp11)
        imloss_center_with_bottomleft_3 = self.comput_imageloss_mean_mask(center_im, centerimg_est_from_bottomleft_3,
                                                                          self.mask_disp12)

        # TOTAL IMAGE LOSS
        return (
                imloss_center_with_topright_1 + imloss_center_with_topright_2 * 2.0 + imloss_center_with_topright_3 * 3.0 +
                imloss_center_with_bottomleft_1 + imloss_center_with_bottomleft_2 * 2.0 + imloss_center_with_bottomleft_3 * 3.0)

    # compute loss among horizental subapertures
    def comput_loss_horizental_orientation(self, center_im, center_disp):
        # get images
        left_im_3 = self.images_list[13]  # 027
        left_im_2 = self.images_list[14]  # 028
        left_im_1 = self.images_list[15]  # 029
        right_im_1 = self.images_list[16]  # 031
        right_im_2 = self.images_list[17]  # 032
        right_im_3 = self.images_list[18]  # 033
        # GENERATE ESTIMATED CENTER IMAGES
        centerimg_est_from_left_1 = self.generate_image_right(left_im_1, center_disp)
        centerimg_est_from_left_2 = self.generate_image_right(left_im_2, center_disp * 2.)
        centerimg_est_from_left_3 = self.generate_image_right(left_im_3, center_disp * 3.)
        centerimg_est_from_right_1 = self.generate_image_left(right_im_1, center_disp)
        centerimg_est_from_right_2 = self.generate_image_left(right_im_2, center_disp * 2.)
        centerimg_est_from_right_3 = self.generate_image_left(right_im_3, center_disp * 3.)

        # GENERATE ESTIMATED LEFT AND RIGHT IMAGES
        # imageListNumber_13_est_form_center = self.generate_image_left(center_im, self.disp_13_est * 3.)
        # imageListNumber_14_est_form_center = self.generate_image_left(center_im, self.disp_14_est * 2.)
        # leftimg_1_est_from_center = self.generate_image_left(center_im, left_disp)
        #
        # rightimg_1_est_from_center = self.generate_image_right(center_im, right_disp)
        # imageListNumber_17_est_form_center = self.generate_image_right(center_im, self.disp_17_est * 2.)
        # imageListNumber_18_est_form_center = self.generate_image_right(center_im, self.disp_18_est * 3.)

        # COMPUTE IMAGE LOSS
        imloss_center_with_left_1 = self.comput_imageloss_mean_mask(center_im, centerimg_est_from_left_1,
                                                                    self.mask_disp15)
        imloss_center_with_left_2 = self.comput_imageloss_mean_mask(center_im, centerimg_est_from_left_2,
                                                                    self.mask_disp14)
        imloss_center_with_left_3 = self.comput_imageloss_mean_mask(center_im, centerimg_est_from_left_3,
                                                                    self.mask_disp13)
        imloss_center_with_right_1 = self.comput_imageloss_mean_mask(center_im, centerimg_est_from_right_1,
                                                                     self.mask_disp16)
        imloss_center_with_right_2 = self.comput_imageloss_mean_mask(center_im, centerimg_est_from_right_2,
                                                                     self.mask_disp17)
        imloss_center_with_right_3 = self.comput_imageloss_mean_mask(center_im, centerimg_est_from_right_3,
                                                                     self.mask_disp18)
        # imloss_right_with_center = self.comput_imageloss_mean(right_im_1, rightimg_1_est_from_center)

        # imloss_imageListNumber_13_with_center = self.comput_imageloss_mean(self.images_list[13],
        #                                                                    imageListNumber_13_est_form_center)
        # imloss_imageListNumber_14_with_center = self.comput_imageloss_mean(self.images_list[14],
        #                                                                    imageListNumber_14_est_form_center)
        # imloss_imageListNumber_17_with_center = self.comput_imageloss_mean(self.images_list[17],
        #                                                                    imageListNumber_17_est_form_center)
        # imloss_imageListNumber_18_with_center = self.comput_imageloss_mean(self.images_list[18],
        #                                                                    imageListNumber_18_est_form_center)

        self.refocusedim_0 = (centerimg_est_from_left_1 + centerimg_est_from_left_2 + centerimg_est_from_left_3 +
                              centerimg_est_from_right_1 + centerimg_est_from_right_2 + centerimg_est_from_right_3)

        return (imloss_center_with_left_1 + imloss_center_with_left_2 * 2.0 + imloss_center_with_left_3 * 3.0 +
                imloss_center_with_right_1 + imloss_center_with_right_2 * 2.0 + imloss_center_with_right_3 * 3.0)

    # compute loss among vertical subapertures
    def comput_loss_vertical_orientation(self, center_im, center_disp):
        # get images
        top_im_3 = self.images_list[19]  # 003
        top_im_2 = self.images_list[20]  # 012
        top_im_1 = self.images_list[21]  # 021
        bottom_im_1 = self.images_list[22]  # 039
        bottom_im_2 = self.images_list[23]  # 048
        bottom_im_3 = self.images_list[24]  # 057
        # GENERATE ESTIMATED CENTER IMAGES
        centerimg_est_from_top_1 = self.generate_image_bottom(top_im_1, center_disp)
        centerimg_est_from_top_2 = self.generate_image_bottom(top_im_2, center_disp * 2.)
        centerimg_est_from_top_3 = self.generate_image_bottom(top_im_3, center_disp * 3.)
        centerimg_est_from_bottom_1 = self.generate_image_top(bottom_im_1, center_disp)
        centerimg_est_from_bottom_2 = self.generate_image_top(bottom_im_2, center_disp * 2.)
        centerimg_est_from_bottom_3 = self.generate_image_top(bottom_im_3, center_disp * 3.)

        # GENERATE ESTIMATED TOP AND BOTTOM IMAGES
        # imageListNumber_19_est_form_center = self.generate_image_top(center_im, self.disp_19_est * 3.)
        # imageListNumber_20_est_form_center = self.generate_image_top(center_im, self.disp_20_est * 2.)
        # topimg_1_est_from_center = self.generate_image_top(center_im, top_disp)
        #
        # bottomimg_1_est_from_center = self.generate_image_bottom(center_im, bottom_disp)
        # imageListNumber_23_est_form_center = self.generate_image_bottom(center_im, self.disp_23_est * 2.)
        # imageListNumber_24_est_form_center = self.generate_image_bottom(center_im, self.disp_24_est * 3.)

        #
        # COMPUTE IMAGE LOSS
        imloss_center_with_top_1 = self.comput_imageloss_mean_mask(center_im, centerimg_est_from_top_1,
                                                                   self.mask_disp21)
        imloss_center_with_top_2 = self.comput_imageloss_mean_mask(center_im, centerimg_est_from_top_2,
                                                                   self.mask_disp20)
        imloss_center_with_top_3 = self.comput_imageloss_mean_mask(center_im, centerimg_est_from_top_3,
                                                                   self.mask_disp19)
        # imloss_top_with_center = self.comput_imageloss_mean(top_im_1, topimg_1_est_from_center)
        imloss_center_with_bottom_1 = self.comput_imageloss_mean_mask(center_im, centerimg_est_from_bottom_1,
                                                                      self.mask_disp22)
        imloss_center_with_bottom_2 = self.comput_imageloss_mean_mask(center_im, centerimg_est_from_bottom_2,
                                                                      self.mask_disp23)
        imloss_center_with_bottom_3 = self.comput_imageloss_mean_mask(center_im, centerimg_est_from_bottom_3,
                                                                      self.mask_disp24)
        # imloss_bottom_with_center = self.comput_imageloss_mean(bottom_im_1, bottomimg_1_est_from_center)

        # imloss_imageListNumber_19_with_center = self.comput_imageloss_mean(self.images_list[19],
        #                                                                    imageListNumber_19_est_form_center)
        # imloss_imageListNumber_20_with_center = self.comput_imageloss_mean(self.images_list[20],
        #                                                                    imageListNumber_20_est_form_center)
        # imloss_imageListNumber_23_with_center = self.comput_imageloss_mean(self.images_list[23],
        #                                                                    imageListNumber_23_est_form_center)
        # imloss_imageListNumber_24_with_center = self.comput_imageloss_mean(self.images_list[24],
        #                                                                    imageListNumber_24_est_form_center)

        self.refocusedim_90 = (centerimg_est_from_top_1 + centerimg_est_from_top_2 + centerimg_est_from_top_3 +
                               centerimg_est_from_bottom_1 + centerimg_est_from_bottom_2 + centerimg_est_from_bottom_3)

        # TOTAL IMAGE LOSS
        return (imloss_center_with_top_1 + imloss_center_with_top_2 * 2.0 + imloss_center_with_top_3 * 3.0 +
                imloss_center_with_bottom_1 + imloss_center_with_bottom_2 * 2.0 + imloss_center_with_bottom_3 * 3.0)

    def comput_loss_45degree_orientation_refine(self, center_im, center_disp):
        # get images
        topleft_im_3 = self.images_list[1]  # 000
        topleft_im_2 = self.images_list[2]  # 010
        topleft_im_1 = self.images_list[3]  # 020
        bottomright_im_1 = self.images_list[4]  # 040
        bottomright_im_2 = self.images_list[5]  # 050
        bottomright_im_3 = self.images_list[6]  # 060

        # GENERATE ESTIMATED CENTER IMAGES
        centerimg_est_from_topleft_1 = self.generate_image_bottomright(topleft_im_1, center_disp, center_disp)
        centerimg_est_from_topleft_2 = self.generate_image_bottomright(topleft_im_2, center_disp * 2., center_disp * 2.)
        centerimg_est_from_topleft_3 = self.generate_image_bottomright(topleft_im_3, center_disp * 3., center_disp * 3.)
        centerimg_est_from_bottomright_1 = self.generate_image_topleft(bottomright_im_1, center_disp, center_disp)
        centerimg_est_from_bottomright_2 = self.generate_image_topleft(bottomright_im_2, center_disp * 2.,
                                                                       center_disp * 2.)
        centerimg_est_from_bottomright_3 = self.generate_image_topleft(bottomright_im_3, center_disp * 3.,
                                                                       center_disp * 3.)
        # COMPUTE IMAGE LOSS
        imloss_center_with_topleft_1 = self.comput_imageloss_mean_mask_refine(center_im,
                                                                              centerimg_est_from_topleft_1,
                                                                              self.mask_disp3)  # if disocc not need
        imloss_center_with_topleft_2 = self.comput_imageloss_mean_mask_refine(center_im, centerimg_est_from_topleft_2,
                                                                              self.mask_disp2)
        imloss_center_with_topleft_3 = self.comput_imageloss_mean_mask_refine(center_im, centerimg_est_from_topleft_3,
                                                                              self.mask_disp1)
        imloss_center_with_bottomright_1 = self.comput_imageloss_mean_mask_refine(center_im,
                                                                                  centerimg_est_from_bottomright_1,
                                                                                  self.mask_disp4)  # if disocc not need
        imloss_center_with_bottomright_2 = self.comput_imageloss_mean_mask_refine(center_im,
                                                                                  centerimg_est_from_bottomright_2,
                                                                                  self.mask_disp5)
        imloss_center_with_bottomright_3 = self.comput_imageloss_mean_mask_refine(center_im,
                                                                                  centerimg_est_from_bottomright_3,
                                                                                  self.mask_disp6)

        self.refocusedim_45 = (
                centerimg_est_from_topleft_1 + centerimg_est_from_topleft_2 + centerimg_est_from_topleft_3 +
                centerimg_est_from_bottomright_1 + centerimg_est_from_bottomright_2 + centerimg_est_from_bottomright_3)

        return (imloss_center_with_topleft_1 + imloss_center_with_topleft_2 * 2.0 + imloss_center_with_topleft_3 * 3.0 +
                imloss_center_with_bottomright_1 + imloss_center_with_bottomright_2 * 2.0 + imloss_center_with_bottomright_3 * 3.0)

    def comput_loss_135degree_orientation_refine(self, center_im, center_disp):
        # get images
        topright_im_3 = self.images_list[7]  # 006
        topright_im_2 = self.images_list[8]  # 014
        topright_im_1 = self.images_list[9]  # 022
        bottomleft_im_1 = self.images_list[10]  # 038
        bottomleft_im_2 = self.images_list[11]  # 046
        bottomleft_im_3 = self.images_list[12]  # 054
        # GENERATE ESTIMATED CENTER IMAGES
        centerimg_est_from_topright_1 = self.generate_image_bottomleft(topright_im_1, center_disp, center_disp)
        centerimg_est_from_topright_2 = self.generate_image_bottomleft(topright_im_2, center_disp * 2.,
                                                                       center_disp * 2.)
        centerimg_est_from_topright_3 = self.generate_image_bottomleft(topright_im_3, center_disp * 3.,
                                                                       center_disp * 3.)
        centerimg_est_from_bottomleft_1 = self.generate_image_topright(bottomleft_im_1, center_disp, center_disp)
        centerimg_est_from_bottomleft_2 = self.generate_image_topright(bottomleft_im_2, center_disp * 2.,
                                                                       center_disp * 2.)
        centerimg_est_from_bottomleft_3 = self.generate_image_topright(bottomleft_im_3, center_disp * 3.,
                                                                       center_disp * 3.)

        # GENERATE ESTIMATED TOPRIGHT AND BOTTOMLEFT IMAGES
        # imageListNumber_7_est_form_center = self.generate_image_topright(center_im, self.disp_7_est * 3.,
        #                                                                  self.disp_7_est * 3.)
        # imageListNumber_8_est_form_center = self.generate_image_topright(center_im, self.disp_8_est * 2.,
        #                                                                  self.disp_8_est * 2.)
        # toprightimg_1_est_from_center = self.generate_image_topright(center_im, topright_disp, topright_disp)
        #
        # bottomleftimg_1_est_from_center = self.generate_image_bottomleft(center_im, bottomleft_disp, bottomleft_disp)
        # imageListNumber_11_est_form_center = self.generate_image_bottomleft(center_im, self.disp_11_est * 2.,
        #                                                                     self.disp_11_est * 2.)
        # imageListNumber_12_est_form_center = self.generate_image_bottomleft(center_im, self.disp_12_est * 3.,
        #                                                                     self.disp_12_est * 3.)

        # COMPUTE IMAGE LOSS
        imloss_center_with_topright_1 = self.comput_imageloss_mean_mask_refine(center_im,
                                                                               centerimg_est_from_topright_1,
                                                                               self.mask_disp9)  # if disocc not need
        imloss_center_with_topright_2 = self.comput_imageloss_mean_mask_refine(center_im, centerimg_est_from_topright_2,
                                                                               self.mask_disp8)
        imloss_center_with_topright_3 = self.comput_imageloss_mean_mask_refine(center_im, centerimg_est_from_topright_3,
                                                                               self.mask_disp7)
        imloss_center_with_bottomleft_1 = self.comput_imageloss_mean_mask_refine(center_im,
                                                                                 centerimg_est_from_bottomleft_1,
                                                                                 self.mask_disp10)  # if disocc not need
        imloss_center_with_bottomleft_2 = self.comput_imageloss_mean_mask_refine(center_im,
                                                                                 centerimg_est_from_bottomleft_2,
                                                                                 self.mask_disp11)
        imloss_center_with_bottomleft_3 = self.comput_imageloss_mean_mask_refine(center_im,
                                                                                 centerimg_est_from_bottomleft_3,
                                                                                 self.mask_disp12)

        # TOTAL IMAGE LOSS
        return (
                imloss_center_with_topright_1 + imloss_center_with_topright_2 * 2.0 + imloss_center_with_topright_3 * 3.0 +
                imloss_center_with_bottomleft_1 + imloss_center_with_bottomleft_2 * 2.0 + imloss_center_with_bottomleft_3 * 3.0)

    # compute loss among horizental subapertures
    def comput_loss_horizental_orientation_refine(self, center_im, center_disp):
        # get images
        left_im_3 = self.images_list[13]  # 027
        left_im_2 = self.images_list[14]  # 028
        left_im_1 = self.images_list[15]  # 029
        right_im_1 = self.images_list[16]  # 031
        right_im_2 = self.images_list[17]  # 032
        right_im_3 = self.images_list[18]  # 033
        # GENERATE ESTIMATED CENTER IMAGES
        centerimg_est_from_left_1 = self.generate_image_right(left_im_1, center_disp)
        centerimg_est_from_left_2 = self.generate_image_right(left_im_2, center_disp * 2.)
        centerimg_est_from_left_3 = self.generate_image_right(left_im_3, center_disp * 3.)
        centerimg_est_from_right_1 = self.generate_image_left(right_im_1, center_disp)
        centerimg_est_from_right_2 = self.generate_image_left(right_im_2, center_disp * 2.)
        centerimg_est_from_right_3 = self.generate_image_left(right_im_3, center_disp * 3.)

        # GENERATE ESTIMATED LEFT AND RIGHT IMAGES
        # imageListNumber_13_est_form_center = self.generate_image_left(center_im, self.disp_13_est * 3.)
        # imageListNumber_14_est_form_center = self.generate_image_left(center_im, self.disp_14_est * 2.)
        # leftimg_1_est_from_center = self.generate_image_left(center_im, left_disp)
        #
        # rightimg_1_est_from_center = self.generate_image_right(center_im, right_disp)
        # imageListNumber_17_est_form_center = self.generate_image_right(center_im, self.disp_17_est * 2.)
        # imageListNumber_18_est_form_center = self.generate_image_right(center_im, self.disp_18_est * 3.)

        # COMPUTE IMAGE LOSS
        imloss_center_with_left_1 = self.comput_imageloss_mean_mask_refine(center_im, centerimg_est_from_left_1,
                                                                           self.mask_disp15)
        imloss_center_with_left_2 = self.comput_imageloss_mean_mask_refine(center_im, centerimg_est_from_left_2,
                                                                           self.mask_disp14)
        imloss_center_with_left_3 = self.comput_imageloss_mean_mask_refine(center_im, centerimg_est_from_left_3,
                                                                           self.mask_disp13)
        imloss_center_with_right_1 = self.comput_imageloss_mean_mask_refine(center_im, centerimg_est_from_right_1,
                                                                            self.mask_disp16)
        imloss_center_with_right_2 = self.comput_imageloss_mean_mask_refine(center_im, centerimg_est_from_right_2,
                                                                            self.mask_disp17)
        imloss_center_with_right_3 = self.comput_imageloss_mean_mask_refine(center_im, centerimg_est_from_right_3,
                                                                            self.mask_disp18)
        # imloss_right_with_center = self.comput_imageloss_mean(right_im_1, rightimg_1_est_from_center)

        # imloss_imageListNumber_13_with_center = self.comput_imageloss_mean(self.images_list[13],
        #                                                                    imageListNumber_13_est_form_center)
        # imloss_imageListNumber_14_with_center = self.comput_imageloss_mean(self.images_list[14],
        #                                                                    imageListNumber_14_est_form_center)
        # imloss_imageListNumber_17_with_center = self.comput_imageloss_mean(self.images_list[17],
        #                                                                    imageListNumber_17_est_form_center)
        # imloss_imageListNumber_18_with_center = self.comput_imageloss_mean(self.images_list[18],
        #                                                                    imageListNumber_18_est_form_center)

        self.refocusedim_0 = (centerimg_est_from_left_1 + centerimg_est_from_left_2 + centerimg_est_from_left_3 +
                              centerimg_est_from_right_1 + centerimg_est_from_right_2 + centerimg_est_from_right_3)

        return (imloss_center_with_left_1 + imloss_center_with_left_2 * 2.0 + imloss_center_with_left_3 * 3.0 +
                imloss_center_with_right_1 + imloss_center_with_right_2 * 2.0 + imloss_center_with_right_3 * 3.0)

    # compute loss among vertical subapertures
    def comput_loss_vertical_orientation_refine(self, center_im, center_disp):
        # get images
        top_im_3 = self.images_list[19]  # 003
        top_im_2 = self.images_list[20]  # 012
        top_im_1 = self.images_list[21]  # 021
        bottom_im_1 = self.images_list[22]  # 039
        bottom_im_2 = self.images_list[23]  # 048
        bottom_im_3 = self.images_list[24]  # 057
        # GENERATE ESTIMATED CENTER IMAGES
        centerimg_est_from_top_1 = self.generate_image_bottom(top_im_1, center_disp)
        centerimg_est_from_top_2 = self.generate_image_bottom(top_im_2, center_disp * 2.)
        centerimg_est_from_top_3 = self.generate_image_bottom(top_im_3, center_disp * 3.)
        centerimg_est_from_bottom_1 = self.generate_image_top(bottom_im_1, center_disp)
        centerimg_est_from_bottom_2 = self.generate_image_top(bottom_im_2, center_disp * 2.)
        centerimg_est_from_bottom_3 = self.generate_image_top(bottom_im_3, center_disp * 3.)

        # GENERATE ESTIMATED TOP AND BOTTOM IMAGES
        # imageListNumber_19_est_form_center = self.generate_image_top(center_im, self.disp_19_est * 3.)
        # imageListNumber_20_est_form_center = self.generate_image_top(center_im, self.disp_20_est * 2.)
        # topimg_1_est_from_center = self.generate_image_top(center_im, top_disp)
        #
        # bottomimg_1_est_from_center = self.generate_image_bottom(center_im, bottom_disp)
        # imageListNumber_23_est_form_center = self.generate_image_bottom(center_im, self.disp_23_est * 2.)
        # imageListNumber_24_est_form_center = self.generate_image_bottom(center_im, self.disp_24_est * 3.)

        #
        # COMPUTE IMAGE LOSS
        imloss_center_with_top_1 = self.comput_imageloss_mean_mask_refine(center_im, centerimg_est_from_top_1,
                                                                          self.mask_disp21)
        imloss_center_with_top_2 = self.comput_imageloss_mean_mask_refine(center_im, centerimg_est_from_top_2,
                                                                          self.mask_disp20)
        imloss_center_with_top_3 = self.comput_imageloss_mean_mask_refine(center_im, centerimg_est_from_top_3,
                                                                          self.mask_disp19)
        # imloss_top_with_center = self.comput_imageloss_mean(top_im_1, topimg_1_est_from_center)
        imloss_center_with_bottom_1 = self.comput_imageloss_mean_mask_refine(center_im, centerimg_est_from_bottom_1,
                                                                             self.mask_disp22)
        imloss_center_with_bottom_2 = self.comput_imageloss_mean_mask_refine(center_im, centerimg_est_from_bottom_2,
                                                                             self.mask_disp23)
        imloss_center_with_bottom_3 = self.comput_imageloss_mean_mask_refine(center_im, centerimg_est_from_bottom_3,
                                                                             self.mask_disp24)
        # imloss_bottom_with_center = self.comput_imageloss_mean(bottom_im_1, bottomimg_1_est_from_center)

        # imloss_imageListNumber_19_with_center = self.comput_imageloss_mean(self.images_list[19],
        #                                                                    imageListNumber_19_est_form_center)
        # imloss_imageListNumber_20_with_center = self.comput_imageloss_mean(self.images_list[20],
        #                                                                    imageListNumber_20_est_form_center)
        # imloss_imageListNumber_23_with_center = self.comput_imageloss_mean(self.images_list[23],
        #                                                                    imageListNumber_23_est_form_center)
        # imloss_imageListNumber_24_with_center = self.comput_imageloss_mean(self.images_list[24],
        #                                                                    imageListNumber_24_est_form_center)

        self.refocusedim_90 = (centerimg_est_from_top_1 + centerimg_est_from_top_2 + centerimg_est_from_top_3 +
                               centerimg_est_from_bottom_1 + centerimg_est_from_bottom_2 + centerimg_est_from_bottom_3)

        # TOTAL IMAGE LOSS
        return (imloss_center_with_top_1 + imloss_center_with_top_2 * 2.0 + imloss_center_with_top_3 * 3.0 +
                imloss_center_with_bottom_1 + imloss_center_with_bottom_2 * 2.0 + imloss_center_with_bottom_3 * 3.0)

    def comput_loss_45degree_center_orientation(self, center_im, center_disp):
        # get images
        img_1 = self.images_list[1]  # 000
        img_2 = self.images_list[2]  # 010
        img_3 = self.images_list[3]  # 020
        img_4 = self.images_list[4]  # 040
        img_5 = self.images_list[5]  # 050
        img_6 = self.images_list[6]  # 060
        img_7 = self.images_list[7]  # 006
        img_8 = self.images_list[8]  # 014
        img_9 = self.images_list[9]  # 022
        img_10 = self.images_list[10]  # 038
        img_11 = self.images_list[11]  # 046
        img_12 = self.images_list[12]  # 054
        img_13 = self.images_list[13]  # 027
        img_14 = self.images_list[14]  # 028
        img_15 = self.images_list[15]  # 029
        img_16 = self.images_list[16]  # 031
        img_17 = self.images_list[17]  # 032
        img_18 = self.images_list[18]  # 033
        img_19 = self.images_list[19]  # 003
        img_20 = self.images_list[20]  # 012
        img_21 = self.images_list[21]  # 021
        img_22 = self.images_list[22]  # 039
        img_23 = self.images_list[23]  # 048
        img_24 = self.images_list[24]  # 057

        # GENERATE ESTIMATED CENTER IMAGES
        wimg_1 = self.generate_image_bottomright(img_1, center_disp * 3., center_disp * 3.)
        wimg_2 = self.generate_image_bottomright(img_2, center_disp * 2., center_disp * 2.)
        wimg_3 = self.generate_image_bottomright(img_3, center_disp, center_disp)
        wimg_4 = self.generate_image_topleft(img_4, center_disp, center_disp)
        wimg_5 = self.generate_image_topleft(img_5, center_disp * 2., center_disp * 2.)
        wimg_6 = self.generate_image_topleft(img_6, center_disp * 3., center_disp * 3.)
        wimg_7 = self.generate_image_bottomleft(img_7, center_disp * 3., center_disp * 3.)
        wimg_8 = self.generate_image_bottomleft(img_8, center_disp * 2., center_disp * 2.)
        wimg_9 = self.generate_image_bottomleft(img_9, center_disp, center_disp)
        wimg_10 = self.generate_image_topright(img_10, center_disp, center_disp)
        wimg_11 = self.generate_image_topright(img_11, center_disp * 2., center_disp * 2.)
        wimg_12 = self.generate_image_topright(img_12, center_disp * 3., center_disp * 3.)
        wimg_13 = self.generate_image_right(img_13, center_disp * 3.)
        wimg_14 = self.generate_image_right(img_14, center_disp * 2.)
        wimg_15 = self.generate_image_right(img_15, center_disp)
        wimg_16 = self.generate_image_left(img_16, center_disp)
        wimg_17 = self.generate_image_left(img_17, center_disp * 2.)
        wimg_18 = self.generate_image_left(img_18, center_disp * 3.)
        wimg_19 = self.generate_image_bottom(img_19, center_disp * 3.)
        wimg_20 = self.generate_image_bottom(img_20, center_disp * 2.)
        wimg_21 = self.generate_image_bottom(img_21, center_disp)
        wimg_22 = self.generate_image_top(img_22, center_disp)
        wimg_23 = self.generate_image_top(img_23, center_disp * 2.)
        wimg_24 = self.generate_image_top(img_24, center_disp * 3.)
        mmean_image_sum = center_im + (wimg_1 * self.mask_disp1) + (wimg_2 * self.mask_disp2) + (
                wimg_3 * self.mask_disp3) + (
                                  wimg_4 * self.mask_disp4) + (wimg_5 * self.mask_disp5) + (wimg_6 * self.mask_disp6) + \
                          (wimg_7 * self.mask_disp7) + (wimg_8 * self.mask_disp8) + (wimg_9 * self.mask_disp9) + (
                                  wimg_10 * self.mask_disp10) + (wimg_11 * self.mask_disp11) + (
                                  wimg_12 * self.mask_disp12) + \
                          (wimg_13 * self.mask_disp13) + (wimg_14 * self.mask_disp14) + (wimg_15 * self.mask_disp15) + (
                                  wimg_16 * self.mask_disp16) + (wimg_17 * self.mask_disp17) + (
                                  wimg_18 * self.mask_disp18) + \
                          (wimg_19 * self.mask_disp19) + (wimg_20 * self.mask_disp20) + (wimg_21 * self.mask_disp21) + (
                                  wimg_22 * self.mask_disp22) + (wimg_23 * self.mask_disp23) + (
                                  wimg_24 * self.mask_disp24)
        self.mask_sum = tf.ones_like(self.mask_disp1, dtype=tf.float32) + \
                        self.mask_disp1 + self.mask_disp2 + self.mask_disp3 + self.mask_disp4 + self.mask_disp5 + self.mask_disp6 + \
                        self.mask_disp7 + self.mask_disp8 + self.mask_disp9 + self.mask_disp10 + self.mask_disp11 + self.mask_disp12 + \
                        self.mask_disp13 + self.mask_disp14 + self.mask_disp15 + self.mask_disp16 + self.mask_disp17 + self.mask_disp18 + \
                        self.mask_disp19 + self.mask_disp20 + self.mask_disp21 + self.mask_disp22 + self.mask_disp23 + self.mask_disp24

        self.mean_image = mmean_image_sum / self.mask_sum

        self.image_consistent_loss = tf.zeros_like(self.mean_image, dtype=tf.float32)
        self.image_consistent_loss += self.comput_l2_loss(center_im, self.mean_image)
        self.image_consistent_loss += self.comput_l2_loss(wimg_1, self.mean_image,
                                                          self.mask_disp1)
        self.image_consistent_loss += self.comput_l2_loss(wimg_2, self.mean_image,
                                                          self.mask_disp2)
        self.image_consistent_loss += self.comput_l2_loss(wimg_3, self.mean_image,
                                                          self.mask_disp3)
        self.image_consistent_loss += self.comput_l2_loss(wimg_4, self.mean_image,
                                                          self.mask_disp4)
        self.image_consistent_loss += self.comput_l2_loss(wimg_5, self.mean_image,
                                                          self.mask_disp5)
        self.image_consistent_loss += self.comput_l2_loss(wimg_6, self.mean_image,
                                                          self.mask_disp6)
        self.image_consistent_loss += self.comput_l2_loss(wimg_7, self.mean_image,
                                                          self.mask_disp7)
        self.image_consistent_loss += self.comput_l2_loss(wimg_8, self.mean_image,
                                                          self.mask_disp8)
        self.image_consistent_loss += self.comput_l2_loss(wimg_9, self.mean_image,
                                                          self.mask_disp9)
        self.image_consistent_loss += self.comput_l2_loss(wimg_10, self.mean_image,
                                                          self.mask_disp10)
        self.image_consistent_loss += self.comput_l2_loss(wimg_11, self.mean_image,
                                                          self.mask_disp11)
        self.image_consistent_loss += self.comput_l2_loss(wimg_12, self.mean_image,
                                                          self.mask_disp12)
        self.image_consistent_loss += self.comput_l2_loss(wimg_13, self.mean_image,
                                                          self.mask_disp13)
        self.image_consistent_loss += self.comput_l2_loss(wimg_14, self.mean_image,
                                                          self.mask_disp14)
        self.image_consistent_loss += self.comput_l2_loss(wimg_15, self.mean_image,
                                                          self.mask_disp15)
        self.image_consistent_loss += self.comput_l2_loss(wimg_16, self.mean_image,
                                                          self.mask_disp16)
        self.image_consistent_loss += self.comput_l2_loss(wimg_17, self.mean_image,
                                                          self.mask_disp17)
        self.image_consistent_loss += self.comput_l2_loss(wimg_18, self.mean_image,
                                                          self.mask_disp18)
        self.image_consistent_loss += self.comput_l2_loss(wimg_19, self.mean_image,
                                                          self.mask_disp19)
        self.image_consistent_loss += self.comput_l2_loss(wimg_20, self.mean_image,
                                                          self.mask_disp20)
        self.image_consistent_loss += self.comput_l2_loss(wimg_21, self.mean_image,
                                                          self.mask_disp21)
        self.image_consistent_loss += self.comput_l2_loss(wimg_22, self.mean_image,
                                                          self.mask_disp22)
        self.image_consistent_loss += self.comput_l2_loss(wimg_23, self.mean_image,
                                                          self.mask_disp23)
        self.image_consistent_loss += self.comput_l2_loss(wimg_24, self.mean_image,
                                                          self.mask_disp24)
        return tf.reduce_mean(self.image_consistent_loss)

    def build_losses(self):
        with tf.variable_scope('losses', reuse=self.reuse_variables):
            # IMAGE LOSS: WEIGTHED SUM

            image_loss_45 = self.comput_loss_45degree_orientation(self.center, self.disp_center_est)
            image_loss_135 = self.comput_loss_135degree_orientation(self.center, self.disp_center_est)
            image_loss_0 = self.comput_loss_horizental_orientation(self.center, self.disp_center_est)
            image_loss_90 = self.comput_loss_vertical_orientation(self.center, self.disp_center_est)
            total_image_loss = image_loss_0 + image_loss_90 + image_loss_45 + image_loss_135

            smooth_loss = self.get_24_smoothlos()
            refine_loss = self.comput_loss_45degree_center_orientation(self.center, self.center_disp)
            image_loss_45 = self.comput_loss_45degree_orientation_refine(self.center, self.center_disp)
            image_loss_135 = self.comput_loss_135degree_orientation_refine(self.center, self.center_disp)
            image_loss_0 = self.comput_loss_horizental_orientation_refine(self.center, self.center_disp)
            image_loss_90 = self.comput_loss_vertical_orientation_refine(self.center, self.center_disp)
            total_image_loss += image_loss_0 + image_loss_90 + image_loss_45 + image_loss_135
            disp_consistent_loss = self.left_right_consistent()
            self.new_loss = self.nine_output(self.center)
            self.imageloss = total_image_loss
            self.total_loss = self.new_loss + self.imageloss + refine_loss * 0.15 + disp_consistent_loss * 0.5 + smooth_loss * 0.05

    def build_summaries(self):
        # SUMMARIES
        with tf.device('/cpu:0'):
            tf.summary.scalar('image_loss', self.imageloss, collections=self.model_collection)
            tf.summary.scalar('new_loss', self.new_loss, collections=self.model_collection)
            tf.summary.image('disp_center_est', (self.disp_center_est + 4.) * 32, max_outputs=4,
                             collections=self.model_collection)
            tf.summary.image('unosleft', (self.mask_disp13_visual) * 32, max_outputs=4,
                             collections=self.model_collection)
            tf.summary.image('unosright', (self.mask_disp18_visual) * 32, max_outputs=4,
                             collections=self.model_collection)

            tf.summary.image('ourleft', (self.mask[13]) * 32, max_outputs=4, collections=self.model_collection)
            tf.summary.image('ourright', (self.mask[18]) * 32, max_outputs=4, collections=self.model_collection)
            tf.summary.image('mixleft', (self.mask_disp13) * 32, max_outputs=4, collections=self.model_collection)
            tf.summary.image('mixright', (self.mask_disp18) * 32, max_outputs=4, collections=self.model_collection)

            if self.params.full_summary:
                tf.summary.image('center_image', self.center, max_outputs=4, collections=self.model_collection)
                tf.summary.image('mask_sum', self.mask_sum, max_outputs=4, collections=self.model_collection)

    def model_summary(self):
        print('\n')
        print('='*30 + 'Model Structure' + '='*30)
        # 获取可训练的variables
        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)
        print('='*60 + '\n')

