# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

"""Monodepth data loader.
"""

from __future__ import absolute_import, division, print_function
import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


def string_length_tf(t):
    return tf.py_func(len, [t], [tf.int64])


class MonodepthDataloader(object):
    """monodepth dataloader"""

    def augment_image_pair(self, left_image, right_image):
        # randomly shift gamma
        random_gamma = tf.random_uniform([], 0.8, 1.2)
        left_image_aug = left_image ** random_gamma
        right_image_aug = right_image ** random_gamma

        # randomly shift brightness
        random_brightness = tf.random_uniform([], 0.5, 2.0)
        left_image_aug = left_image_aug * random_brightness
        right_image_aug = right_image_aug * random_brightness

        # randomly shift color
        random_colors = tf.random_uniform([3], 0.8, 1.2)
        white = tf.ones([tf.shape(left_image)[0], tf.shape(left_image)[1]])
        color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=2)
        left_image_aug *= color_image
        right_image_aug *= color_image

        # saturate
        left_image_aug = tf.clip_by_value(left_image_aug, 0, 1)
        right_image_aug = tf.clip_by_value(right_image_aug, 0, 1)

        return left_image_aug, right_image_aug

    def augment_image_pair_list(self, image_list):
        # randomly shift gamma
        random_gamma = tf.random_uniform([], 0.8, 1.2)
        # left_image_aug = left_image ** random_gamma
        # right_image_aug = right_image ** random_gamma

        image_aug_list = [single ** random_gamma for single in image_list]

        # randomly shift brightness
        random_brightness = tf.random_uniform([], 0.5, 2.0)
        # left_image_aug = left_image_aug * random_brightness
        # right_image_aug = right_image_aug * random_brightness
        image_aug_list = [single * random_brightness for single in image_aug_list]

        # randomly shift color
        random_colors = tf.random_uniform([3], 0.8, 1.2)
        white = tf.ones([tf.shape(image_list[0])[0], tf.shape(image_list[0])[1]])
        color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=2)
        # left_image_aug *= color_image
        # right_image_aug *= color_image
        image_aug_list = [single * color_image for single in image_aug_list]

        # saturate
        # left_image_aug = tf.clip_by_value(left_image_aug, 0, 1)
        # right_image_aug = tf.clip_by_value(right_image_aug, 0, 1)
        image_aug_list = [tf.clip_by_value(single, 0, 1) for single in image_aug_list]

        return image_aug_list

    def augment_image_pair4(self, center_image, top_image, bottom_image, left_image, right_image):
        # randomly shift gamma
        random_gamma = tf.random_uniform([], 0.8, 1.2)
        center_image_aug = center_image ** random_gamma
        top_image_aug = top_image ** random_gamma
        bottom_image_aug = bottom_image ** random_gamma
        left_image_aug = left_image ** random_gamma
        right_image_aug = right_image ** random_gamma

        # randomly shift brightness
        random_brightness = tf.random_uniform([], 0.8, 1.2)
        center_image_aug = center_image_aug * random_brightness
        top_image_aug = top_image_aug * random_brightness
        bottom_image_aug = bottom_image_aug * random_brightness
        left_image_aug = left_image_aug * random_brightness
        right_image_aug = right_image_aug * random_brightness

        # randomly shift color
        random_colors = tf.random_uniform([3], 0.8, 1.2)
        white = tf.ones([tf.shape(center_image)[0], tf.shape(center_image)[1]])
        color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=2)
        center_image_aug *= color_image
        top_image_aug *= color_image
        bottom_image_aug *= color_image
        left_image_aug *= color_image
        right_image_aug *= color_image

        # saturate
        center_image_aug = tf.clip_by_value(center_image_aug, 0, 1)
        top_image_aug = tf.clip_by_value(top_image_aug, 0, 1)
        bottom_image_aug = tf.clip_by_value(bottom_image_aug, 0, 1)
        left_image_aug = tf.clip_by_value(left_image_aug, 0, 1)
        right_image_aug = tf.clip_by_value(right_image_aug, 0, 1)

        return center_image_aug, top_image_aug, bottom_image_aug, left_image_aug, right_image_aug

    def augment_image_pair9(self, topLeft_image, top_image, topRight_image,
                            left_image, center_image, right_image,
                            bottomLeft_image, bottom_image, bottomRight_image):
        def random_scaling(im):
            in_h, in_w, _ = im.get_shape().as_list()
            scaling = tf.random_uniform([2], 1, 1.15)
            x_scaling = scaling[0]
            y_scaling = scaling[1]
            out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
            out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)
            im = tf.image.resize_area(im, [out_h, out_w])
            return im

        def random_cropping(im, out_h, out_w):
            # batch_size, in_h, in_w, _ = im.get_shape().as_list()
            in_h, in_w, _ = tf.unstack(tf.shape(im))
            offset_y = tf.random_uniform([1], 0, in_h - out_h + 1, dtype=tf.int32)[0]
            offset_x = tf.random_uniform([1], 0, in_w - out_w + 1, dtype=tf.int32)[0]
            im = tf.image.crop_to_bounding_box(
                im, offset_y, offset_x, out_h, out_w)
            return im

        def geoaugment(img):
            # imgaugment = random_scaling(img)
            imgaugment = random_cropping(img, 512, 512)
            # imgaugment = tf.cast(imgaugment, dtype=tf.uint8)
            return imgaugment

        # geonet dataaugment

        # topLeft_image = geoaugment(topLeft_image)
        # top_image = geoaugment(top_image)
        # topRight_image = geoaugment(topRight_image)
        #
        # left_image = geoaugment(left_image)
        # center_image = geoaugment(center_image)
        # right_image = geoaugment(right_image)
        #
        # bottomLeft_image = geoaugment(bottomLeft_image)
        # bottom_image = geoaugment(bottom_image)
        # bottomRight_image = geoaugment(bottomRight_image)

        # randomly shift gamma
        random_gamma = tf.random_uniform([], 0.8, 1.2)

        topLeft_image_aug = topLeft_image ** random_gamma
        top_image_aug = top_image ** random_gamma
        topRight_image_aug = topRight_image ** random_gamma

        left_image_aug = left_image ** random_gamma
        center_image_aug = center_image ** random_gamma
        right_image_aug = right_image ** random_gamma

        bottomLeft_image_aug = bottomLeft_image ** random_gamma
        bottom_image_aug = bottom_image ** random_gamma
        bottomRight_image_aug = bottomRight_image ** random_gamma

        # randomly shift brightness

        random_brightness = tf.random_uniform([], 0.8, 1.2)  # 0.5 - 2.0

        topLeft_image_aug = topLeft_image_aug * random_brightness
        top_image_aug = top_image_aug * random_brightness
        topRight_image_aug = topRight_image_aug * random_brightness

        left_image_aug = left_image_aug * random_brightness
        center_image_aug = center_image_aug * random_brightness
        right_image_aug = right_image_aug * random_brightness

        bottomLeft_image_aug = bottomLeft_image_aug * random_brightness
        bottom_image_aug = bottom_image_aug * random_brightness
        bottomRight_image_aug = bottomRight_image_aug * random_brightness

        # convert to color scale
        # flog =  tf.image.rgb_to_grayscale(bottomLeft_image_aug)

        # randomly shift color
        random_colors = tf.random_uniform([3], 0.5, 2)  # 0.8-1.2
        white = tf.ones([tf.shape(center_image)[0], tf.shape(center_image)[1]])
        color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=2)

        topLeft_image_aug *= color_image
        top_image_aug *= color_image
        topRight_image_aug *= color_image

        left_image_aug *= color_image
        center_image_aug *= color_image
        right_image_aug *= color_image

        bottomLeft_image_aug *= color_image
        bottom_image_aug *= color_image
        bottomRight_image_aug *= color_image

        # saturate
        topLeft_image_aug = tf.clip_by_value(topLeft_image_aug, 0, 1)
        top_image_aug = tf.clip_by_value(top_image_aug, 0, 1)
        topRight_image_aug = tf.clip_by_value(topRight_image_aug, 0, 1)

        left_image_aug = tf.clip_by_value(left_image_aug, 0, 1)
        center_image_aug = tf.clip_by_value(center_image_aug, 0, 1)
        right_image_aug = tf.clip_by_value(right_image_aug, 0, 1)

        bottomLeft_image_aug = tf.clip_by_value(bottomLeft_image_aug, 0, 1)
        bottom_image_aug = tf.clip_by_value(bottom_image_aug, 0, 1)
        bottomRight_image_aug = tf.clip_by_value(bottomRight_image_aug, 0, 1)

        return topLeft_image_aug, top_image_aug, topRight_image_aug, left_image_aug, center_image_aug, right_image_aug, bottomLeft_image_aug, bottom_image_aug, bottomRight_image_aug

    def augment_image_pair_2dimensions(self, center_image, verticle_image, horizonal_image):
        """
        created by zec
        :param center_image:
        :param verticle_image:
        :param horizonal_image:
        :return: three input after augment
        """
        # randomly shift gamma
        random_gamma = tf.random_uniform([], 0.8, 1.2)
        center_image_aug = center_image ** random_gamma
        verticle_image_aug = [d ** random_gamma for d in verticle_image]
        horizonal_image_aug = [d ** random_gamma for d in horizonal_image]

        # randomly shift brightness
        random_brightness = tf.random_uniform([], 0.5, 2.0)
        center_image_aug = center_image_aug * random_brightness
        verticle_image_aug = [d * random_brightness for d in verticle_image_aug]
        horizonal_image_aug = [d * random_brightness for d in horizonal_image_aug]

        # randomly shift color
        random_colors = tf.random_uniform([3], 0.8, 1.2)
        white = tf.ones([tf.shape(center_image)[0], tf.shape(center_image)[1]])
        color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=2)
        center_image_aug *= color_image
        verticle_image_aug = [d * color_image for d in verticle_image_aug]
        horizonal_image_aug = [d * color_image for d in horizonal_image_aug]

        # saturate
        center_image_aug = tf.clip_by_value(center_image_aug, 0, 1)
        verticle_image_aug = [tf.clip_by_value(d, 0, 1) for d in verticle_image_aug]
        horizonal_image_aug = [tf.clip_by_value(d, 0, 1) for d in horizonal_image_aug]

        return center_image_aug, verticle_image_aug, horizonal_image_aug

    # def augment_image_pair_1dimensions(self, input_image,probability_gamma,probability_brightness,probability_shift_color,probability_saturate):
    #     """
    #     :param input_image:
    #     :param probability_gamma:
    #     :param probability_brightness:
    #     :param probability_shift_color:
    #     :param probability_saturate:
    #     :return:
    #     """
    #     # randomly shift gamma
    #     random_gamma = tf.random_uniform([], 0.8, 1.2)
    #     input_image_aug  = input_image  ** random_gamma
    #     verticle_image_aug = [d ** random_gamma for d in verticle_image]
    #     horizonal_image_aug = [d ** random_gamma for d in horizonal_image]
    #
    #
    #     # randomly shift brightness
    #     random_brightness = tf.random_uniform([], 0.5, 2.0)
    #     center_image_aug  =  center_image_aug * random_brightness
    #     verticle_image_aug = [d * random_brightness for d in verticle_image_aug]
    #     horizonal_image_aug = [d * random_brightness for d in horizonal_image_aug]
    #
    #     # randomly shift color
    #     random_colors = tf.random_uniform([3], 0.8, 1.2)
    #     white = tf.ones([tf.shape(center_image)[0], tf.shape(center_image)[1]])
    #     color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=2)
    #     center_image_aug  *= color_image
    #     verticle_image_aug = [d *color_image for d in verticle_image_aug]
    #     horizonal_image_aug = [d *color_image for d in horizonal_image_aug]
    #
    #     # saturate
    #     center_image_aug  = tf.clip_by_value(center_image_aug,  0, 1)
    #     verticle_image_aug = [tf.clip_by_value(d,  0, 1) for d in verticle_image_aug]
    #     horizonal_image_aug = [tf.clip_by_value(d,  0, 1) for d in horizonal_image_aug]
    #
    #     return center_image_aug, verticle_image_aug,horizonal_image_aug

    def __init__(self, data_path, filenames_file, params, mode):
        self.data_path = data_path
        self.params = params
        # self.dataset = dataset
        self.mode = mode

        # self.center_image_batch = None
        # self.topLeft_image_batch = None
        # self.top_image_batch = None
        # self.topRight_image_batch = None
        # self.left_image_batch = None
        # self.right_image_batch = None
        # self.bottomLeft_image_batch = None
        # self.bottom_image_batch = None
        # self.bottomRight_image_batch = None
        self.image_batch_list = None

        input_queue = tf.train.string_input_producer([filenames_file], shuffle=False)
        line_reader = tf.TextLineReader()
        _, line = line_reader.read(input_queue)

        split_line = tf.string_split([line]).values

        # we load only one image for test, except if we trained a stereo model
        if mode == 'test':  # and not self.params.do_stereo:
            # center_image_path  = tf.string_join([self.data_path,'/', split_line[0]])
            # center_image_o  = self.read_image(center_image_path)

            center_image_path = [tf.string_join([self.data_path, '/', split_line[i]]) for i in range(25)]
            center_image_o = [self.read_image(singel) for singel in center_image_path]

        elif mode == 'test_flipud':
            # center_image_path = tf.string_join([self.data_path, '/', split_line[0]])
            # center_image_o = self.read_image(center_image_path)
            #
            center_image_path = [tf.string_join([self.data_path, '/', split_line[i]]) for i in range(25)]
            center_image_o = [self.read_image(singel) for singel in center_image_path]
        #
        elif mode == 'test_flip_up_lr':
            # center_image_path = tf.string_join([self.data_path, '/', split_line[0]])
            # center_image_o = self.read_image(center_image_path)
            #
            center_image_path = [tf.string_join([self.data_path, '/', split_line[i]]) for i in range(25)]
            center_image_o = [self.read_image(singel) for singel in center_image_path]
        else:

            # center_image_path = tf.string_join([self.data_path, '/',split_line[0]])
            #
            # topLeft_image_path = tf.string_join([self.data_path, '/', split_line[1]])
            # top_image_path = tf.string_join([self.data_path,'/', split_line[2]])
            # topRight_image_path = tf.string_join([self.data_path, '/', split_line[3]])
            #
            # left_image_path = tf.string_join([self.data_path, '/', split_line[4]])
            # right_image_path = tf.string_join([self.data_path, '/', split_line[5]])
            #
            # bottomLeft_image_path = tf.string_join([self.data_path, '/', split_line[6]])
            # bottom_image_path = tf.string_join([self.data_path,'/', split_line[7]])
            # bottomRight_image_path = tf.string_join([self.data_path, '/', split_line[8]])

            images_path_list = [tf.string_join([self.data_path, '/', split_line[i]]) for i in range(25)]

            # center_image_o = self.read_image(center_image_path)
            #
            # topLeft_image_o = self.read_image(topLeft_image_path)
            # top_image_o =self.read_image(top_image_path)
            # topRight_image_o = self.read_image(topRight_image_path)
            #
            # left_image_o = self.read_image(left_image_path)
            # right_image_o = self.read_image(right_image_path)
            #
            # bottomLeft_image_o = self.read_image(bottomLeft_image_path)
            # bottom_image_o =self.read_image(bottom_image_path)
            # bottomRight_image_o = self.read_image(bottomRight_image_path)

            image_o_list = [self.read_image(singel) for singel in images_path_list]

        if mode == 'train':
            # randomly flip images
            def list_reverse(mylist):
                mylist.reverse()
                return mylist

            do_flip = tf.random_uniform([], 0, 1)
            do_flipud = tf.random_uniform([], 0, 1)

            # topLeft_image = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(topRight_image_o), lambda: topLeft_image_o)
            # top_image = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(top_image_o), lambda: top_image_o)
            # topRight_image = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(topLeft_image_o),lambda: topRight_image_o)
            #
            # left_image = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(right_image_o), lambda: left_image_o)
            # center_image = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(center_image_o),lambda: center_image_o)
            # right_image = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(left_image_o), lambda: right_image_o)
            #
            # bottomLeft_image = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(bottomRight_image_o), lambda: bottomLeft_image_o)
            # bottom_image = tf.cond(do_flip >0.5, lambda: tf.image.flip_left_right(bottom_image_o), lambda: bottom_image_o)
            # bottomRight_image = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(bottomLeft_image_o), lambda: bottomRight_image_o)
            imagecenter_o = image_o_list[0]
            imagena_o = image_o_list[1:7]  # \            get 1, 2, 3, 4, 5, 6
            imagepie_o = image_o_list[7:13]  # /           get 7, 8, 9, 10,11,12
            imageheng_o = image_o_list[13:19]  # --        get 13,14,15,16,17,18
            imageheng_o_inv = image_o_list[18:12:-1]  # -- get 18,17,16,15,14,13
            imageshu_o = image_o_list[19:25]  # |          get 19,20,21,22,23,24

            imagecenter = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(imagecenter_o), lambda: imagecenter_o)
            imagena = tf.cond(do_flip > 0.5, lambda: [tf.image.flip_left_right(single) for single in imagepie_o],
                              lambda: imagena_o)
            imagepie = tf.cond(do_flip > 0.5, lambda: [tf.image.flip_left_right(single) for single in imagena_o],
                               lambda: imagepie_o)
            imageheng = tf.cond(do_flip > 0.5, lambda: [tf.image.flip_left_right(single) for single in imageheng_o_inv],
                                lambda: imageheng_o)
            imageshu = tf.cond(do_flip > 0.5, lambda: [tf.image.flip_left_right(single) for single in imageshu_o],
                               lambda: imageshu_o)

            """
            def list_reverse(image_list):
                image_list.reverse()
                return image_list
            flag = tf.cond(do_flip > 0.5, lambda: list_reverse(imageheng_o), lambda: imageheng_o)
            imageheng = tf.cond(do_flip > 0.5, lambda: [tf.image.flip_left_right(single)for single in imageheng_o], lambda: imageheng_o)
            """

            # random flip up down
            imagena_inv = imagena[::-1]
            imagepie_inv = imagepie[::-1]
            imageshu_inv = imageshu[::-1]
            imagecenter = tf.cond(do_flipud > 0.5, lambda: tf.image.flip_up_down(imagecenter), lambda: imagecenter)
            imagena = tf.cond(do_flipud > 0.5, lambda: [tf.image.flip_up_down(single) for single in imagepie_inv],
                              lambda: imagena)
            imagepie = tf.cond(do_flipud > 0.5, lambda: [tf.image.flip_up_down(single) for single in imagena_inv],
                               lambda: imagepie)
            imageheng = tf.cond(do_flipud > 0.5, lambda: [tf.image.flip_up_down(single) for single in imageheng],
                                lambda: imageheng)
            imageshu = tf.cond(do_flipud > 0.5, lambda: [tf.image.flip_up_down(single) for single in imageshu_inv],
                               lambda: imageshu)

            # images = [imagecenter]+imagepie+imagena+imageheng+imageshu
            images = [imagecenter] + imagena + imagepie + imageheng + imageshu

            # randomly augment images
            do_augment = tf.random_uniform([], 0, 0.4)

            # topLeft_image,top_image,topRight_image,left_image,center_image,right_image,bottomLeft_image,bottom_image,bottomRight_image = tf.cond(do_augment > 0.5,
            #                               lambda: self.augment_image_pair9(topLeft_image,top_image,topRight_image,left_image,center_image,right_image,bottomLeft_image,bottom_image,bottomRight_image),
            #                               lambda: (topLeft_image,top_image,topRight_image,left_image,center_image,right_image,bottomLeft_image,bottom_image,bottomRight_image))

            images = tf.cond(do_augment > 0.5, lambda: self.augment_image_pair_list(images), lambda: images)

            # topLeft_image.set_shape([None, None, 3])
            # top_image.set_shape([None, None, 3])
            # topRight_image.set_shape([None, None, 3])
            #
            # left_image.set_shape([None, None, 3])
            # center_image.set_shape([None, None, 3])
            # right_image.set_shape([None, None, 3])
            #
            # bottomLeft_image.set_shape([None, None, 3])
            # bottom_image.set_shape([None, None, 3])
            # bottomRight_image.set_shape([None, None, 3])
            for single in images:
                single.set_shape([None, None, 3])

            min_after_dequeue = 63
            capacity = min_after_dequeue + 4 * params.batch_size

            # self.topLeft_image_batch,self.top_image_batch,self.topRight_image_batch,self.left_image_batch, self.center_image_batch,self.right_image_batch, \
            # self.bottomLeft_image_batch,self.bottom_image_batch,self.bottomRight_image_batch  = tf.train.shuffle_batch(
            #     [topLeft_image, top_image,topRight_image,left_image,center_image, right_image,bottomLeft_image,bottom_image, bottomRight_image],
            #     params.batch_size, capacity, min_after_dequeue, params.num_threads)
            self.image_batch_list = tf.train.shuffle_batch(images, params.batch_size, capacity, min_after_dequeue,
                                                           params.num_threads)

        elif mode == 'test':
            # self.left_image_batch = tf.stack([left_image_o,  tf.image.flip_left_right(left_image_o)],  0)
            # self.left_image_batch.set_shape( [2, None, None, 3])
            do_flip = tf.random_uniform([], 0.6, 1)
            imagecenter_o = center_image_o[0]
            imagena_o = center_image_o[1:7]  # \            get 1, 2, 3, 4, 5, 6
            imagepie_o = center_image_o[7:13]  # /           get 7, 8, 9, 10,11,12
            imageheng_o = center_image_o[13:19]  # --        get 13,14,15,16,17,18
            imageheng_o_inv = center_image_o[18:12:-1]  # -- get 18,17,16,15,14,13
            imageshu_o = center_image_o[19:25]  # |          get 19,20,21,22,23,24

            imagecenter = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(imagecenter_o), lambda: imagecenter_o)
            imagena = tf.cond(do_flip > 0.5, lambda: [tf.image.flip_left_right(single) for single in imagepie_o],
                              lambda: imagena_o)
            imagepie = tf.cond(do_flip > 0.5, lambda: [tf.image.flip_left_right(single) for single in imagena_o],
                               lambda: imagepie_o)
            imageheng = tf.cond(do_flip > 0.5, lambda: [tf.image.flip_left_right(single) for single in imageheng_o_inv],
                                lambda: imageheng_o)
            imageshu = tf.cond(do_flip > 0.5, lambda: [tf.image.flip_left_right(single) for single in imageshu_o],
                               lambda: imageshu_o)

            flip_lr = [imagecenter] + imagena + imagepie + imageheng + imageshu

            #
            # flip_lr = []  # 25 ( 512, 512, 3)
            # for single in center_image_o:
            #     single = tf.image.flip_left_right(single)
            #     flip_lr.append(single)

            self.center_image_batch = []
            for i in range(25):
                temp = tf.stack([center_image_o[i], flip_lr[i]], axis=0)
                temp.set_shape([2, None, None, 3])
                self.center_image_batch.append(temp)

            # self.center_image_batch = tf.stack([center_image_o, tf.image.flip_left_right(center_image_o)], 0)
            # self.center_image_batch.set_shape([2, None, None, 3])

            # ( 4, 512, 512, 3)

        elif mode == 'test_flipud':
            do_flip = tf.random_uniform([], 0, 0.4)
            do_flipud = tf.random_uniform([], 0.6, 1)
            imagecenter_o = center_image_o[0]
            imagena_o = center_image_o[1:7]  # \            get 1, 2, 3, 4, 5, 6
            imagepie_o = center_image_o[7:13]  # /           get 7, 8, 9, 10,11,12
            imageheng_o = center_image_o[13:19]  # --        get 13,14,15,16,17,18
            imageheng_o_inv = center_image_o[18:12:-1]  # -- get 18,17,16,15,14,13
            imageshu_o = center_image_o[19:25]  # |          get 19,20,21,22,23,24

            imagecenter = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(imagecenter_o), lambda: imagecenter_o)
            imagena = tf.cond(do_flip > 0.5, lambda: [tf.image.flip_left_right(single) for single in imagepie_o],
                              lambda: imagena_o)
            imagepie = tf.cond(do_flip > 0.5, lambda: [tf.image.flip_left_right(single) for single in imagena_o],
                               lambda: imagepie_o)
            imageheng = tf.cond(do_flip > 0.5, lambda: [tf.image.flip_left_right(single) for single in imageheng_o_inv],
                                lambda: imageheng_o)
            imageshu = tf.cond(do_flip > 0.5, lambda: [tf.image.flip_left_right(single) for single in imageshu_o],
                               lambda: imageshu_o)

            # random flip up down
            imagena_inv = imagena[::-1]
            imagepie_inv = imagepie[::-1]
            imageshu_inv = imageshu[::-1]
            imagecenter = tf.cond(do_flipud > 0.5, lambda: tf.image.flip_up_down(imagecenter), lambda: imagecenter)
            imagena = tf.cond(do_flipud > 0.5, lambda: [tf.image.flip_up_down(single) for single in imagepie_inv],
                              lambda: imagena)
            imagepie = tf.cond(do_flipud > 0.5, lambda: [tf.image.flip_up_down(single) for single in imagena_inv],
                               lambda: imagepie)
            imageheng = tf.cond(do_flipud > 0.5, lambda: [tf.image.flip_up_down(single) for single in imageheng],
                                lambda: imageheng)
            imageshu = tf.cond(do_flipud > 0.5, lambda: [tf.image.flip_up_down(single) for single in imageshu_inv],
                               lambda: imageshu)

            # images = [imagecenter]+imagepie+imagena+imageheng+imageshu
            flip_ud = [imagecenter] + imagena + imagepie + imageheng + imageshu

            # flip_ud = []  # 25 ( 512, 512, 3)
            # for single in center_image_o:
            #     single = tf.image.flip_up_down(single)
            #     flip_ud.append(single)

            self.center_image_batch = []
            for i in range(25):
                temp = tf.stack([center_image_o[i], flip_ud[i]], axis=0)
                temp.set_shape([2, None, None, 3])
                self.center_image_batch.append(temp)
            

            # self.center_image_batch = tf.stack([center_image_o, tf.image.flip_up_down(center_image_o)], 0)
            # self.center_image_batch.set_shape([2, None, None, 3])
        elif mode == 'test_flip_up_lr':

            do_flip = tf.random_uniform([], 0.6, 1.0)
            not_do_fliplr = tf.random_uniform([], 0, 0.4)
            do_flipud = tf.random_uniform([], 0.6, 1.0)

            imagecenter_o = center_image_o[0]
            imagena_o = center_image_o[1:7]  # \            get 1, 2, 3, 4, 5, 6
            imagepie_o = center_image_o[7:13]  # /           get 7, 8, 9, 10,11,12
            imageheng_o = center_image_o[13:19]  # --        get 13,14,15,16,17,18
            imageheng_o_inv = center_image_o[18:12:-1]  # -- get 18,17,16,15,14,13
            imageshu_o = center_image_o[19:25]  # |          get 19,20,21,22,23,24

            imagecenter = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(imagecenter_o), lambda: imagecenter_o)
            imagena = tf.cond(do_flip > 0.5, lambda: [tf.image.flip_left_right(single) for single in imagepie_o],
                              lambda: imagena_o)
            imagepie = tf.cond(do_flip > 0.5, lambda: [tf.image.flip_left_right(single) for single in imagena_o],
                               lambda: imagepie_o)
            imageheng = tf.cond(do_flip > 0.5, lambda: [tf.image.flip_left_right(single) for single in imageheng_o_inv],
                                lambda: imageheng_o)
            imageshu = tf.cond(do_flip > 0.5, lambda: [tf.image.flip_left_right(single) for single in imageshu_o],
                               lambda: imageshu_o)

            flip_lr = [imagecenter] + imagena + imagepie + imageheng + imageshu

            # random flip up down
            imagecenter = tf.cond(not_do_fliplr > 0.5, lambda: tf.image.flip_left_right(imagecenter_o),
                                  lambda: imagecenter_o)
            imagena = tf.cond(not_do_fliplr > 0.5, lambda: [tf.image.flip_left_right(single) for single in imagepie_o],
                              lambda: imagena_o)
            imagepie = tf.cond(not_do_fliplr > 0.5, lambda: [tf.image.flip_left_right(single) for single in imagena_o],
                               lambda: imagepie_o)
            imageheng = tf.cond(not_do_fliplr > 0.5,
                                lambda: [tf.image.flip_left_right(single) for single in imageheng_o_inv],
                                lambda: imageheng_o)
            imageshu = tf.cond(not_do_fliplr > 0.5, lambda: [tf.image.flip_left_right(single) for single in imageshu_o],
                               lambda: imageshu_o)

            imagena_inv = imagena[::-1]
            imagepie_inv = imagepie[::-1]
            imageshu_inv = imageshu[::-1]
            imagecenter = tf.cond(do_flipud > 0.5, lambda: tf.image.flip_up_down(imagecenter), lambda: imagecenter)
            imagena = tf.cond(do_flipud > 0.5, lambda: [tf.image.flip_up_down(single) for single in imagepie_inv],
                              lambda: imagena)
            imagepie = tf.cond(do_flipud > 0.5, lambda: [tf.image.flip_up_down(single) for single in imagena_inv],
                               lambda: imagepie)
            imageheng = tf.cond(do_flipud > 0.5, lambda: [tf.image.flip_up_down(single) for single in imageheng],
                                lambda: imageheng)
            imageshu = tf.cond(do_flipud > 0.5, lambda: [tf.image.flip_up_down(single) for single in imageshu_inv],
                               lambda: imageshu)

            flip_ud = [imagecenter] + imagena + imagepie + imageheng + imageshu

            self.center_image_batch = []
            for i in range(25):
                temp = tf.stack([center_image_o[i], flip_lr[i], flip_ud[i]], axis=0)
                temp.set_shape([3, None, None, 3])
                self.center_image_batch.append(temp)

            # self.center_image_batch = tf.stack(
            #     [center_image_o, tf.image.flip_left_right(center_image_o), tf.image.flip_up_down(center_image_o)], 0)
            #
            # self.center_image_batch.set_shape([3, None, None, 3])
            # if self.params.do_stereo:
            #    self.right_image_batch = tf.stack([right_image_o,  tf.image.flip_left_right(right_image_o)],  0)
            #    self.right_image_batch.set_shape( [2, None, None, 3])

    def read_image(self, image_path):
        # tf.decode_image does not return the image size, this is an ugly workaround to handle both jpeg and png
        # path_length = string_length_tf(image_path)[0]
        # file_extension = tf.substr(image_path, path_length - 3, 3)
        # file_cond = tf.equal(file_extension, 'jpg')
        file_cond = tf.convert_to_tensor(False, dtype=tf.bool)
        image = tf.cond(file_cond, lambda: tf.image.decode_jpeg(tf.read_file(image_path)),
                        lambda: tf.image.decode_png(tf.read_file(image_path)))

        # if the dataset is cityscapes, we crop the last fifth to remove the car hood

        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_images(image, [self.params.height, self.params.width], tf.image.ResizeMethod.AREA)
        # subtract mean create by zec
        # image = tf.subtract(image,tf.reduce_mean(image))
        return image
