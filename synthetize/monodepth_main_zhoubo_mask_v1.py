# -*- coding:UTF-8 -*-
# Copyright UCL B45000usiness plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

from __future__ import absolute_import, division, print_function
import os


# only keep warnings and errors

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import argparse
from monodepth_model_zhoubo_mask_v1 import *
from tensorflow.python import pywrap_tensorflow
import cv2
from dataloader_zhoubo_mask_v1 import *
from average_gradients import *
from evalfunctions7x7 import *

# traing:0--------------------test:1----------------get_all_pfm:2
train_or_test = 0
parser = argparse.ArgumentParser(description='4D LF Monodepth with TensorFlow implementation.')
filenames_file = '/root/Algorithm/UnLFdisp/synthetize/4dlf_7x7star_add_train.txt'  # train  filename_file path
filenames_fileTest = '/root/Algorithm/UnLFdisp/synthetize/4dlffilenames_7x7star_test.txt'  # test or val

dir = os.getcwd()
curdir = "/".join(dir.split("/")[:-1])
gt_path = curdir + '/evaluation_toolkit/data/eval_gt'
data_path = curdir + "/evaluation_toolkit/data/full_data"
#
# gt_path = "/home/fufu/data/eval_gt"
# data_path = "/home/fufu/data"

output_directory = '/root/Algorithm/UnLFdisp/synthetize/result-alpha-0.90'  # save the test result into this dir
log_directory = output_directory
checkpoint_path = ''
# checkpoint_path = '/root/Algorithm/UnLFdisp/synthetize/ckpt_94.10/model-19350'
# checkpoint_path = '/home/fufu/work/code_UnLFdisp/synthetize/result/monoLFdepth/model-80000'

parser.add_argument('--model_name', type=str, help='model name', default='monoLFdepth')
parser.add_argument('--data_path', type=str, help='path to the data', required=False, default=data_path)

parser.add_argument('--input_height', type=int, help='input height', default=512)
parser.add_argument('--input_width', type=int, help='input width', default=512)
parser.add_argument('--batch_size', type=int, help='batch size', default=1)
parser.add_argument('--num_epochs', type=int, help='number of epochs', default=450 * 3)  # 450*3
parser.add_argument('--learning_rate', type=float, help='initial learning rate', default=1e-4)

parser.add_argument('--dp_consistency_sigmoid_scale', type=float,
                    help='scale for sigmoid function in dp_consist computation', default=1.)  # old100
parser.add_argument('--gradient_img_scale', type=float, help='scale for gx and gy', default=0.4)

parser.add_argument('--alpha_image_loss', type=float, help='weight between SSIM and L1 in the image loss', default=0.90)
parser.add_argument('--disp_consistency_loss_weight', type=float, help='left-right consistency weight',
                    default=0.001)  # 0.001
parser.add_argument('--disp_gradient_loss_weight', type=float, help='disparity smoothness weigth', default=0.0)
parser.add_argument('--centerSymmetry_loss_weight', type=float, help='left-center-right consistency weight', default=1.)

parser.add_argument('--use_deconv', help='if set, will use transposed convolutions', action='store_true')
parser.add_argument('--num_gpus', type=int, help='number of GPUs to use for training', default=1)
parser.add_argument('--num_threads', type=int, help='number of threads to use for data loading', default=4)
parser.add_argument('--output_directory', type=str,
                    help='output directory for test disparities, if empty outputs to checkpoint folder',
                    default=output_directory)
parser.add_argument('--log_directory', type=str, help='directory to save checkpoints and summaries',
                    default=log_directory)

if train_or_test == 0:
    if not os.path.exists(log_directory):
        os.mkdir(log_directory)
    parser.add_argument('--mode', type=str, help='train or test', default='train')
    parser.add_argument('--filenames_file', type=str, help='path to the filenames text file', required=False,
                        default=filenames_file)
    parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load',
                        default=checkpoint_path)
    parser.add_argument('--retrain', help='if used with checkpoint_path, will restart training from step zero',
                        action='store_true', default=True)
    parser.add_argument('--full_summary',
                        help='if set, will keep more data for each summary. Warning: the file can become very large',
                        action='store_true', default=True)
elif train_or_test == 1:
    parser.add_argument('--mode', type=str, help='train or test', default='test')
    parser.add_argument('--filenames_file', type=str, help='path to the filenames text file', required=False,
                        default=filenames_fileTest)  # filenames_fileTest
    parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load',
                        default=checkpoint_path)
    parser.add_argument('--retrain', help='if used with checkpoint_path, will restart training from step zero',
                        action='store_true')
    parser.add_argument('--full_summary',
                        help='if set, will keep more data for each summary. Warning: the file can become very large',
                        action='store_true')
elif train_or_test == 2:
    parser.add_argument('--mode', type=str, help='train or test', default='getall')
    parser.add_argument('--filenames_file', type=str, help='path to the filenames text file', required=False,
                        default=filenames_file)
    parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load',
                        default=checkpoint_path)
    parser.add_argument('--retrain', help='if used with checkpoint_path, will restart training from step zero',
                        action='store_true', default=True)
    parser.add_argument('--full_summary',
                        help='if set, will keep more data for each summary. Warning: the file can become very large',
                        action='store_true', default=True)
args = parser.parse_args()


def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def post_process_disparity_flipud(disp):
    _, h, w = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.flipud(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w))
    l = np.rot90(l, -1)
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.flipud(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def count_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def load_param_only(sess, ckpt_path):
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
    print("load success")
    # reader = tf.train.NewCheckpointReader(checkpoint_path)
    restore_dict = dict()
    # print("1111111111111111111111111111111")
    for v in tf.trainable_variables():
        tensor_name = v.name.split(':')[0]
        if reader.has_tensor(tensor_name):
            # print(reader.get_tensor(tensor_name).shape)
            print('has tensor', tensor_name)
            restore_dict[tensor_name] = v
            # print(v.shape)
    # print("2222222222222222222222222222222")
    saver = tf.train.Saver(restore_dict)
    # print("3333333333333333333333333333333")
    saver.restore(sess, ckpt_path)
    # print("4444444444444444444444444444444")
    return True


def train(params):
    """Training loop."""
    # with tf.Graph().as_default(), tf.device('/gpu:0'):
    with tf.Graph().as_default(), tf.device('/gpu:0'):
        s = time.time()
        global_step = tf.Variable(0, trainable=False)

        # OPTIMIZER
        num_training_samples = count_text_lines(args.filenames_file)

        steps_per_epoch = np.ceil(num_training_samples / params.batch_size).astype(np.int32)
        num_total_steps = params.num_epochs * steps_per_epoch

        boundaries = [np.int32(10000), np.int32(20000),
                      np.int32(30000), np.int32(40000), np.int32(165000), np.int32(205000)]

        values = [args.learning_rate, args.learning_rate / 2, args.learning_rate / 4, args.learning_rate / 8,
                  args.learning_rate / 16, args.learning_rate / 32, args.learning_rate / 64]

        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

        print("total number of samples: {}".format(num_training_samples))
        print("total number of steps: {}".format(num_total_steps))

        # Optimizer
        opt_step = tf.train.AdamOptimizer(learning_rate)

        # loading data
        dataloader = MonodepthDataloader(args.data_path, args.filenames_file, params, args.mode)

        images_list = dataloader.image_batch_list
        images_splits_list = [tf.split(single, args.num_gpus, 0) for single in images_list]

        tower_grads = []
        tower_losses = []
        reuse_variables = None
        print("time_cpu:", time.time() - s)
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(args.num_gpus):
                with tf.device('/gpu:%d' % i):
                    images_splits = [single[i] for single in images_splits_list]
                    # print("111111111111")
                    model = MonodepthModel(params, args.mode, images_splits, reuse_variables=reuse_variables,
                                           model_index=i)

                    loss = model.total_loss
                    if (len(tower_losses) != 0):
                        tower_losses = tower_losses.append(loss)
                    elif len(tower_losses) == 0:
                        tower_losses.append(loss)

                    reuse_variables = True

                    grads = opt_step.compute_gradients(loss)
                    new_grads = []
                    for tvar in grads:
                        if tvar[0] is not None:
                            new_grads.append(tvar)
                    tower_grads.append(new_grads)

        grads = average_gradients(tower_grads)

        apply_gradient_op = opt_step.apply_gradients(grads, global_step=global_step)

        total_loss = tf.reduce_mean(tower_losses)

        tf.summary.scalar('learning_rate', learning_rate, ['model_0'])
        tf.summary.scalar('total_loss', total_loss, ['model_0'])
        summary_op = tf.summary.merge_all('model_0')

        # SESSION
        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)

        # SAVER
        summary_writer = tf.summary.FileWriter(args.log_directory + '/' + args.model_name, sess.graph)
        train_saver = tf.train.Saver(max_to_keep=40)

        # COUNT PARAMS
        total_num_parameters = 0
        for variable in tf.trainable_variables():
            total_num_parameters += np.array(variable.get_shape().as_list()).prod()
        print("number of trainable parameters: {}".format(total_num_parameters))

        # INIT
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        # LOAD CHECKPOINT IF SET
        if args.checkpoint_path != '':

            # train_saver.restore(sess, args.checkpoint_path)

            load_success_or_not = load_param_only(sess, checkpoint_path)
            if load_success_or_not == True:
                print('load model successful')

            if args.retrain:
                sess.run(global_step.assign(1))
                # print ('retrain , assign 0')

        # GO!
        start_step = global_step.eval(session=sess)
        start_time = time.time()
        temp = 1.0
        flag = 1
        for step in range(start_step, 300000):
            # if flag:
            #     best_score = eval_all_up_lr(sess, step, 'test_flip_up_lr')
            #     flag = 0
            before_op_time = time.time()
            _, loss_value, single_loss, loss_list = sess.run([apply_gradient_op, total_loss, loss, tower_losses])
            duration = time.time() - before_op_time
            # print(duration)
            if step and step % 100 == 0:
                examples_per_sec = params.batch_size / duration
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / step - 1.0) * time_sofar
                print_string = 'batch {:>6} | examples/s: {:4.2f} | loss: {:.5f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(print_string.format(step, examples_per_sec, loss_value, time_sofar, training_time_left))
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, global_step=step)
            # if step and step % 100 == 0:
            #     score = eval_all_up_lr(sess, step, 'test_flip_up_lr')
            #     if score < best_score:
            #         best_score = score
            #         train_saver.save(sess, args.log_directory + '/' + args.model_name + '/model', global_step=step)
            if step and step % 5000 == 0:
                train_saver.save(sess, args.log_directory + '/' + args.model_name + '/model', global_step=step)
                eval_all_up_lr(sess, step, 'test_flip_up_lr')
            # if single_loss < temp:
            #     print("存在更小的loss-------------------------{:.5f}".format(loss_value))
            #     print("存在更小的single loss-------------------------{:.5f}".format(single_loss))
            #     temp = single_loss
            #     train_saver.save(sess, args.log_directory + '/' + args.model_name + '/model', global_step=step)
            #     eval_all_up_lr(sess, step, 'test_flip_up_lr')
        #     if step == 233887:
        #         eval_all_up_lr(sess, step, 'test_flip_up_lr')
        #         break
        # train_saver.save(sess, args.log_directory + '/' + args.model_name + '/model', global_step=num_total_steps)

        print('done.')


# def get_all_pfm(params, times):
#     """Test function."""
#     mode = "test_flip_up_lr"
#     evaldataloader = MonodepthDataloader(args.data_path, filenames_fileTest, params, mode)
#     center_image = evaldataloader.center_image_batch
#     model = MonodepthModel(params, "test_flip_up_lr", center_image,
#                            reuse_variables=False, model_index=None)
#
#     # SESSION
#     config = tf.ConfigProto(allow_soft_placement=True)
#     sess = tf.Session(config=config)
#
#     # SAVER
#     train_saver = tf.train.Saver()
#
#     # INIT
#     sess.run(tf.global_variables_initializer())
#     sess.run(tf.local_variables_initializer())
#     load_success_or_not = load_param_only(sess, checkpoint_path)
#     if load_success_or_not == True:
#         print('load model successful')
#     coordinator = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)
#
#     num_test_samples = count_text_lines(filenames_fileTest)
#
#     print('now testing {} files'.format(num_test_samples))
#     disparities = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
#     disparities_pp = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
#
#     disparities0 = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
#     disparities_pp0 = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
#     img_disp = np.zeros([28, 512, 512, 24])
#     img_mask = np.zeros([28, 512, 512, 24])
#     # img_our_mask = np.zeros([28, 512, 512, 24])
#
#     for step in range(num_test_samples):
#         two_disp, out_mask, out_disp = sess.run([model.two_centerdisp, model.our_mask, model.all_disp])
#         disp = two_disp[0]
#         disp0 = two_disp[1]
#         img_disp[step, ...] = out_disp[0, ...]
#         img_mask[step, ...] = out_mask[0, :, :, 1:]
#
#         disparities[step] = disp[0].squeeze()
#         disparities0[step] = disp0[0].squeeze()
#
#         if mode == "test_flipud":
#             disparities_pp[step] = post_process_disparity_flipud(disp.squeeze())
#             disparities_pp0[step] = post_process_disparity_flipud(disp0.squeeze())
#         elif mode == "test_flip_up_lr":
#             pp_up = post_process_disparity(disp[0:2, :, :].squeeze())
#             temp_disp = np.expand_dims(disp[2:, :, :].squeeze(), axis=0)
#             pp_up_result = np.concatenate((np.expand_dims(pp_up, axis=0), temp_disp), axis=0)
#             disparities_pp[step] = post_process_disparity_flipud(pp_up_result.squeeze())
#
#             pp_up0 = post_process_disparity(disp0[0:2, :, :].squeeze())
#             temp_disp0 = np.expand_dims(disp0[2:, :, :].squeeze(), axis=0)
#             pp_up_result0 = np.concatenate((np.expand_dims(pp_up0, axis=0), temp_disp0), axis=0)
#             disparities_pp0[step] = post_process_disparity_flipud(pp_up_result0.squeeze())
#
#         else:
#             disparities_pp[step] = post_process_disparity(disp.squeeze())
#
#             disparities_pp0[step] = post_process_disparity(disp0.squeeze())
#
#     print('done.')
#
#     print('writing disparities.')
#     if args.output_directory == '':
#         output_directory = os.path.dirname(args.checkpoint_path)
#     else:
#         output_directory = args.output_directory
#     np.save(output_directory + '/disparities.npy', disparities)
#
#     myresult = open(output_directory + '/result.txt', 'a+')
#     testfile = open(filenames_fileTest)
#     print("-----------------load checkpoint {}---------------".format(str(times)))
#     myresult.write("-----------------load checkpoint {}".format(str(times) + "\n"))
#     if mode == "test_flipud":
#         print("pp_flipud")
#         myresult.write("pp_flipud" + "\n")
#     elif mode == "test":
#         print("pp_fliplr")
#         myresult.write("pp_fliplr" + "\n")
#     elif mode == "test_flip_up_lr":
#         print("test_flip_up_lr")
#         myresult.write("test_flip_up_lr" + "\n")
#     t = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
#     front_name = "/inputCam_{}_our.png"
#     pfm_name = "/{}.pfm"
#     for i in range(disparities.shape[0]):
#         fileline = testfile.readline().split('/')
#         firstclass = fileline[0]
#         scenename = fileline[1]
#         num_id = fileline[2][9:12]
#         whole_name = scenename + "_" + num_id
#         if not os.path.exists(output_directory + "/" + str(times)):
#             os.mkdir(output_directory + "/" + str(times))
#         evalpath = output_directory + "/" + str(times)
#         if not os.path.exists(evalpath + "/" + scenename):
#             os.mkdir(evalpath + "/" + scenename)
#         # write_pfm(disparities_pp[i, :, :], evalpath + '/' + scenename + "/" + scenename + '.pfm')
#         for k in range(24):
#             cv2.imwrite(evalpath + "/" + scenename + front_name.format(str(t[k]).zfill(3)),
#                         (img_mask[i, :, :, k] * 255.0).astype(np.uint8))
#         # for k in range(24):
#         #     write_pfm(img_disp[i, :, :, k], evalpath + "/" + scenename + pfm_name.format(str(t[k]).zfill(3)))
#     print('done.')



def get_all_pfm(params, times):
    """Test function."""
    mode = "test_flip_up_lr"
    evaldataloader = MonodepthDataloader(args.data_path, filenames_file, params, mode)
    center_image = evaldataloader.center_image_batch
    model = MonodepthModel(params, "test_flip_up_lr", center_image,
                           reuse_variables=False, model_index=None)

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # SAVER
    train_saver = tf.train.Saver()

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    load_success_or_not = load_param_only(sess, checkpoint_path)
    if load_success_or_not == True:
        print('load model successful')
    # print("111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111")
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    num_test_samples = count_text_lines(filenames_file)

    print('now testing {} files'.format(num_test_samples))
    disparities = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    disparities_pp = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)

    disparities0 = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    disparities_pp0 = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    img_disp = np.zeros([28, 512, 512, 24])
    img_mask = np.zeros([28, 512, 512, 24])

    for step in range(num_test_samples):
        # s = time.time()
        two_disp = sess.run(model.two_centerdisp)
        # print(time.time() - s)
        disp = two_disp[0]
        disp0 = two_disp[1]
        # img_disp[step, ...] = out_disp[0, ...]
        # img_mask[step, ...] = out_mask[0, ...]

        disparities[step] = disp[0].squeeze()
        disparities0[step] = disp0[0].squeeze()

        if mode == "test_flipud":
            disparities_pp[step] = post_process_disparity_flipud(disp.squeeze())
            disparities_pp0[step] = post_process_disparity_flipud(disp0.squeeze())
        elif mode == "test_flip_up_lr":
            pp_up = post_process_disparity(disp[0:2, :, :].squeeze())
            temp_disp = np.expand_dims(disp[2:, :, :].squeeze(), axis=0)
            pp_up_result = np.concatenate((np.expand_dims(pp_up, axis=0), temp_disp), axis=0)
            disparities_pp[step] = post_process_disparity_flipud(pp_up_result.squeeze())

            pp_up0 = post_process_disparity(disp0[0:2, :, :].squeeze())
            temp_disp0 = np.expand_dims(disp0[2:, :, :].squeeze(), axis=0)
            pp_up_result0 = np.concatenate((np.expand_dims(pp_up0, axis=0), temp_disp0), axis=0)
            disparities_pp0[step] = post_process_disparity_flipud(pp_up_result0.squeeze())

        else:
            disparities_pp[step] = post_process_disparity(disp.squeeze())

            disparities_pp0[step] = post_process_disparity(disp0.squeeze())

    print('done.')

    print('writing disparities.')
    if args.output_directory == '':
        output_directory = os.path.dirname(args.checkpoint_path)
    else:
        output_directory = args.output_directory
    np.save(output_directory + '/disparities.npy', disparities)

    myresult = open(output_directory + '/result.txt', 'a+')
    testfile = open(filenames_file)
    print("-----------------load checkpoint {}---------------".format(str(times)))
    myresult.write("-----------------load checkpoint {}".format(str(times) + "\n"))
    if mode == "test_flipud":
        print("pp_flipud")
        myresult.write("pp_flipud" + "\n")
    elif mode == "test":
        print("pp_fliplr")
        myresult.write("pp_fliplr" + "\n")
    elif mode == "test_flip_up_lr":
        print("test_flip_up_lr")
        myresult.write("test_flip_up_lr" + "\n")
    t = [1, 2, 3, 5, 6, 7, 10, 11, 12, 14, 15, 16, 19, 20, 21, 23, 24, 25, 28, 29, 30, 32, 33, 34]
    front_name = "/inputCam_{}.png"
    pfm_name = "/{}.pfm"
    for i in range(disparities.shape[0]):
        fileline = testfile.readline().split('/')
        firstclass = fileline[1]
        scenename = fileline[2]
        num_id = fileline[3][9:12]
        whole_name = scenename + "_" + num_id
        if not os.path.exists(output_directory + "/" + str(times)):
            os.mkdir(output_directory + "/" + str(times))
        evalpath = output_directory + "/" + str(times)
        # if not os.path.exists(evalpath + "/" + scenename):
        #     os.mkdir(evalpath + "/" + scenename)
        write_pfm(disparities_pp[i, :, :], evalpath + '/' + scenename +
                  '.pfm')
        # for k in range(24):
        #     cv2.imwrite(evalpath + "/" + scenename + front_name.format(str(t[k]).zfill(3)),
        #                 (img_mask[i, :, :, k] * 255.0).astype(np.uint8))
        # for k in range(24):
        #     write_pfm(img_disp[i, :, :, k], evalpath + "/" + scenename + pfm_name.format(str(t[k]).zfill(3)))
    print('done.')


def test_pp(params, times):
    """Test function."""
    mode = "test_flip_up_lr"
    evaldataloader = MonodepthDataloader(args.data_path, filenames_fileTest, params, mode)
    center_image = evaldataloader.center_image_batch
    model = MonodepthModel(params, "test_flip_up_lr", center_image,
                           reuse_variables=False, model_index=None)

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # SAVER
    train_saver = tf.train.Saver()

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    load_success_or_not = load_param_only(sess, checkpoint_path)
    if load_success_or_not == True:
        print('load model successful')
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    num_test_samples = count_text_lines(filenames_fileTest)

    print('now testing {} files'.format(num_test_samples))
    img = np.zeros((num_test_samples, 8, params.height, params.width, 1), dtype=np.float32)
    disparities = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    disparities_pp = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)

    disparities0 = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    disparities_pp0 = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)

    for step in range(num_test_samples):
        two_disp = sess.run(model.two_centerdisp)
        # disp, disp0 = sess.run([model.center_disp, model.center_disp])
        disp = two_disp[0]
        disp0 = two_disp[1]

        disparities[step] = disp[0].squeeze()
        disparities0[step] = disp0[0].squeeze()

        if mode == "test_flipud":
            disparities_pp[step] = post_process_disparity_flipud(disp.squeeze())
            disparities_pp0[step] = post_process_disparity_flipud(disp0.squeeze())
        elif mode == "test_flip_up_lr":
            pp_up = post_process_disparity(disp[0:2, :, :].squeeze())
            temp_disp = np.expand_dims(disp[2:, :, :].squeeze(), axis=0)
            pp_up_result = np.concatenate((np.expand_dims(pp_up, axis=0), temp_disp), axis=0)
            disparities_pp[step] = post_process_disparity_flipud(pp_up_result.squeeze())

            pp_up0 = post_process_disparity(disp0[0:2, :, :].squeeze())
            temp_disp0 = np.expand_dims(disp0[2:, :, :].squeeze(), axis=0)
            pp_up_result0 = np.concatenate((np.expand_dims(pp_up0, axis=0), temp_disp0), axis=0)
            disparities_pp0[step] = post_process_disparity_flipud(pp_up_result0.squeeze())

        else:
            disparities_pp[step] = post_process_disparity(disp.squeeze())

            disparities_pp0[step] = post_process_disparity(disp0.squeeze())

    print('done.')

    print('writing disparities.')
    if args.output_directory == '':
        output_directory = os.path.dirname(args.checkpoint_path)
    else:
        output_directory = args.output_directory
    np.save(output_directory + '/disparities.npy', disparities)

    avg_score = 0.
    avg_score_pp = 0.

    avg_score0 = 0.
    avg_score_pp0 = 0.
    myresult = open(output_directory + '/result.txt', 'a+')
    testfile = open(filenames_fileTest)
    print("-----------------load checkpoint {}---------------".format(str(times)))
    myresult.write("-----------------load checkpoint {}".format(str(times) + "\n"))
    if mode == "test_flipud":
        print("pp_flipud")
        myresult.write("pp_flipud" + "\n")
    elif mode == "test":
        print("pp_fliplr")
        myresult.write("pp_fliplr" + "\n")
    elif mode == "test_flip_up_lr":
        print("test_flip_up_lr")
        myresult.write("test_flip_up_lr" + "\n")
    for i in range(disparities.shape[0]):

        fileline = testfile.readline().split('/')
        firstclass = fileline[0]
        scenename = fileline[1]

        print("origin result")
        if not os.path.exists(output_directory + "/" + str(times)):
            os.mkdir(output_directory + "/" + str(times))
        evalpath = output_directory + "/" + str(times)
        gt = os.path.join(gt_path, scenename + "/valid_disp_map.npy")
        if not os.path.exists(gt):
            print("error path " + gt)
        gt_img = np.load(gt)

        print("-----------------disp1---------------")
        myresult.write("-----------------disp1 {}---------------".format('\n'))
        error_img, error_score = get_scores_file_by_name(disparities[i, :, :], scenename, myresult)
        # write_pfm(disparities[i, :, :], evalpath + '/' + scenename + '.pfm')

        save_erroplt_by_name(gt_img, disparities[i, :, :], error_img, evalpath, scenename, 0)
        avg_score += error_score

        print("-----------------disp1pp---------------")
        myresult.write("-----------------disp1pp {}---------------".format('\n'))
        # get pp result
        error_img, error_score = get_scores_file_by_name(disparities_pp[i, :, :], scenename, myresult)
        write_pfm(disparities_pp[i, :, :], evalpath + '/' + scenename + '.pfm')
        save_erroplt_by_name(gt_img, disparities_pp[i, :, :], error_img, evalpath, scenename, 1)
        avg_score_pp += error_score

        print("-----------------disp0---------------")
        myresult.write("-----------------disp0 {}---------------".format('\n'))
        error_img0, error_score0 = get_scores_file_by_name(disparities0[i, :, :], scenename, myresult)
        save_erroplt_by_name(gt_img, disparities0[i, :, :], error_img0, evalpath, scenename, 2)
        avg_score0 += error_score0

        print("-----------------disp0---------------")
        myresult.write("-----------------disp0pp {}---------------".format('\n'))
        error_img0, error_score0 = get_scores_file_by_name(disparities_pp0[i, :, :], scenename, myresult)
        save_erroplt_by_name(gt_img, disparities_pp0[i, :, :], error_img0, evalpath, scenename, 3)
        avg_score_pp0 += error_score0

        #
        # write_pfm(disparities[i, :, :], output_directory + '/' + VAL_IMAGES[i] + '_dis1.pfm')

        save_erroplt(gt_img, disparities_pp[i, :, :], error_img, output_directory, i, True)
    myresult.write("-----------------avg score {}".format(str(100 - avg_score / 8) + '\n'))
    print("-----------------avg score {}".format(str(100 - avg_score / 8)))
    myresult.write("-----------------avg score {}".format(str(100 - avg_score_pp / 8) + '\n'))
    print("-----------------avg score {}".format(str(100 - avg_score_pp / 8)))

    myresult.write("-----------------avg score {}".format(str(100 - avg_score0 / 8) + '\n'))
    print("-----------------avg score {}".format(str(100 - avg_score0 / 8)))
    myresult.write("-----------------avg score {}".format(str(100 - avg_score_pp0 / 8) + '\n'))
    print("-----------------avg score {}".format(str(100 - avg_score_pp0 / 8)))

    # get_scores(disparities_pp[i, :, :], i)

    print('done.')


def eval_all_up_lr(sess, times, mode):
    """eval function."""
    params = monodepth_parameters(
        height=args.input_height,
        width=args.input_width,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        num_epochs=args.num_epochs,
        use_deconv=args.use_deconv,
        alpha_image_loss=args.alpha_image_loss,
        dp_consistency_sigmoid_scale=args.dp_consistency_sigmoid_scale,
        # gradient_scale=args.gradient_img_scale,
        disp_gradient_loss_weight=args.disp_gradient_loss_weight,
        centerSymmetry_loss_weight=args.centerSymmetry_loss_weight,
        disp_consistency_loss_weight=args.disp_consistency_loss_weight,
        full_summary=args.full_summary)

    evaldataloader = MonodepthDataloader(args.data_path, filenames_fileTest, params, mode)
    center_image = evaldataloader.center_image_batch
    model = MonodepthModel(params, "test_flip_up_lr", center_image,
                           reuse_variables=True, model_index=None)

    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    num_test_samples = count_text_lines(filenames_fileTest)

    disparities = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    disparities_pp = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)

    disparities0 = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    disparities_pp0 = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)

    for step in range(num_test_samples):
        two_disp = sess.run(model.two_centerdisp)
        disp = two_disp[0]
        disp0 = two_disp[1]

        disparities[step] = disp[0].squeeze()
        disparities0[step] = disp0[0].squeeze()

        if mode == "test_flipud":
            disparities_pp[step] = post_process_disparity_flipud(disp.squeeze())
            disparities_pp0[step] = post_process_disparity_flipud(disp0.squeeze())
        elif mode == "test_flip_up_lr":
            pp_up = post_process_disparity(disp[0:2, :, :].squeeze())
            temp_disp = np.expand_dims(disp[2:, :, :].squeeze(), axis=0)
            pp_up_result = np.concatenate((np.expand_dims(pp_up, axis=0), temp_disp), axis=0)
            disparities_pp[step] = post_process_disparity_flipud(pp_up_result.squeeze())

            pp_up0 = post_process_disparity(disp0[0:2, :, :].squeeze())
            temp_disp0 = np.expand_dims(disp0[2:, :, :].squeeze(), axis=0)
            pp_up_result0 = np.concatenate((np.expand_dims(pp_up0, axis=0), temp_disp0), axis=0)
            disparities_pp0[step] = post_process_disparity_flipud(pp_up_result0.squeeze())

        else:
            disparities_pp[step] = post_process_disparity(disp.squeeze())

            disparities_pp0[step] = post_process_disparity(disp0.squeeze())
    if args.output_directory == '':
        output_directory = os.path.dirname(args.checkpoint_path)
    else:
        output_directory = args.output_directory
    np.save(output_directory + '/disparities.npy', disparities)
    avg_score = 0.
    avg_score_pp = 0.

    avg_score0 = 0.
    avg_score_pp0 = 0.
    myresult = open(output_directory + '/result.txt', 'a+')
    testfile = open(filenames_fileTest)
    # print("-----------------load checkpoint {}---------------".format(str(times)))
    myresult.write("-----------------load checkpoint {}".format(str(times) + "\n"))
    if mode == "test_flipud":
        # print("pp_flipud")
        myresult.write("pp_flipud" + "\n")
    elif mode == "test":
        # print("pp_fliplr")
        myresult.write("pp_fliplr" + "\n")
    elif mode == "test_flip_up_lr":
        # print("test_flip_up_lr")
        myresult.write("test_flip_up_lr" + "\n")
    for i in range(disparities.shape[0]):

        fileline = testfile.readline().split('/')
        firstclass = fileline[0]
        scenename = fileline[1]

        # print("origin result")
        if not os.path.exists(output_directory + "/" + str(times)):
            os.mkdir(output_directory + "/" + str(times))
        evalpath = output_directory + "/" + str(times)

        gt = os.path.join(gt_path, scenename + "/valid_disp_map.npy")
        if not os.path.exists(gt):
            print("error path " + gt)
        gt_img = np.load(gt)

        # print("-----------------disp1---------------")
        myresult.write("-----------------disp1 {}---------------".format('\n'))
        error_img, error_score = get_scores_file_by_name_clear(disparities[i, :, :], scenename, myresult)
        write_pfm(disparities[i, :, :], evalpath + '/' + scenename + '_dis.pfm')

        save_erroplt_by_name(gt_img, disparities[i, :, :], error_img, evalpath, scenename, 0)
        avg_score += error_score

        # print("-----------------disp1pp---------------")
        myresult.write("-----------------disp1pp {}---------------".format('\n'))
        # get pp result
        error_img, error_score = get_scores_file_by_name_clear(disparities_pp[i, :, :], scenename, myresult)
        write_pfm(disparities_pp[i, :, :], evalpath + '/' + scenename + '_PPdis.pfm')
        save_erroplt_by_name(gt_img, disparities_pp[i, :, :], error_img, evalpath, scenename, 1)
        avg_score_pp += error_score

        # print("-----------------disp0---------------")
        myresult.write("-----------------disp0 {}---------------".format('\n'))
        error_img0, error_score0 = get_scores_file_by_name_clear(disparities0[i, :, :], scenename, myresult)
        save_erroplt_by_name(gt_img, disparities0[i, :, :], error_img0, evalpath, scenename, 2)
        avg_score0 += error_score0

        # print("-----------------disp0---------------")
        myresult.write("-----------------disp0pp {}---------------".format('\n'))
        error_img0, error_score0 = get_scores_file_by_name_clear(disparities_pp0[i, :, :], scenename, myresult)
        save_erroplt_by_name(gt_img, disparities_pp0[i, :, :], error_img0, evalpath, scenename, 3)
        avg_score_pp0 += error_score0
    myresult.write("-----------------avg score {}".format(str(100 - avg_score / 8) + '\n'))
    print("-----------------avg score {}".format(str(100 - avg_score / 8)))
    myresult.write("-----------------avg score {}".format(str(100 - avg_score_pp / 8) + '\n'))
    print("-----------------avg score {}".format(str(100 - avg_score_pp / 8)))

    myresult.write("-----------------avg score {}".format(str(100 - avg_score0 / 8) + '\n'))
    print("-----------------avg score {}".format(str(100 - avg_score0 / 8)))
    myresult.write("-----------------avg score {}".format(str(100 - avg_score_pp0 / 8) + '\n'))
    print("-----------------avg score {}".format(str(100 - avg_score_pp0 / 8)))

    # get_scores(disparities_pp[i, :, :], i)
    # print(times, "-----------------avg score {}".format(str(100 - avg_score_pp / 8) + '\n'))
    # print('done.')
    return avg_score_pp


def main(_):
    params = monodepth_parameters(
        height=args.input_height,
        width=args.input_width,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        num_epochs=args.num_epochs,
        use_deconv=args.use_deconv,
        alpha_image_loss=args.alpha_image_loss,
        dp_consistency_sigmoid_scale=args.dp_consistency_sigmoid_scale,
        # gradient_scale=args.gradient_img_scale,
        disp_gradient_loss_weight=args.disp_gradient_loss_weight,
        centerSymmetry_loss_weight=args.centerSymmetry_loss_weight,
        disp_consistency_loss_weight=args.disp_consistency_loss_weight,
        full_summary=args.full_summary)

    if args.mode == 'train':
        train(params)
    elif args.mode == 'test':
        # test_full(params)
        test_pp(params, 100000)
    else:
        get_all_pfm(params, 8)


if __name__ == '__main__':
    tf.app.run()
