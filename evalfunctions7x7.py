import sys
import os
sys.path.append("..")
import evaluation_toolkit.source.toolkit.utils.misc as my_misc
import evaluation_toolkit.source.toolkit.settings as setting
from random import *
import numpy as np
import eval_tools
from evaluate import *
from evaluation_toolkit.source.toolkit.metrics.general_metrics import BadPix
import matplotlib
from matplotlib import pyplot as plt

VAL_IMAGES = [
    "sideboard", "cotton", "boxes", "dino",
    # "antinous","greek","dishes","tower",
    "backgammon", "pyramids", "stripes", "dots",
    # "rosemary","boardgames","museum","pillows","tomb","vinyl",
    # "kitchen","medieval2","pens","platonic","table","town",
    # 'buddha', 'buddha2', 'monasRoom', 'papillon', 'stillLife'
]
# VAL_IMAGES = ['sideboard', 'cotton', 'boxes', 'dino', 'backgammon', 'pyramids', 'stripes', 'dots', 'XcEq0gc4UOiS4I', 'z1DefSIynpJhqi', 'PJStXHnbRIFySM', 'WQtPxzYjybY7mQ', 'FRMYJ3bYIKVICq', 'kAC6iBTa8WGyQj', 'BrZmxtWCIkYTFU', 'jsf5J74mdQ6tRN', 'trMltdlvXzRdOS', 'F9oJj8EUagULX3', '2puzMNJK1wuyUX', 'iR3hQ5zpr9Uyhu', '6CgoBrTon07emN', 'ZrQu9KDhI8RUu5', 'GumNhefYrATJLh', '0AruXjjpWdmTOz', 'DcO5nAshBnldAx', 'UPArlHLz4rPfcV', 'cTRQYxjW6XXw5J', 'iq16JtRgF7yzKp', 'Uj2OOu6EgqFWXV', 'Z3wMEwUQMaLPIH', 'a1dUOWBAZyuzkO', 'd0zweKDB7m2oMH', 'v0Hzj8f2extfoJ', 'SUzHVCKCCm6Ax0', 'X0uSv5ZI4rHDyW', 'rf12q5PRLxUPZ7', 'g3CxzfVmydmYGr', 'nTal0eYLxsPnD0', 'gZ392ME3DDQPeX', 'iYypMkXHvDoSPc', 'XMOaUbcV6MWPah', 'xcSnmjlK4lMfOG', 'KS12tXJxSZEcOC', 'XmafNPgVLAHh8A', 'ApjUMjuSV66Jg9', 'ZAlpH5bIkWwbvJ', 'xPnYKUqIpQSZUv', 'WIeKGtXWKXHCYu', '0x8IbwzW484CWq', 'SM3dQt479k0pHI', 'UklDlFYwRcgipI', 'kqK4J4cafLswf2', 'kXmKkyCBJaq0Ku', 'hYhxN330S5JZHe', 'Hn7zY895MqgY2n', 'qPS9zDEjhwzIez', 'QnR1RnBziGiwrV', 'PD5jyKuAaHolM4', 'JhtAbaf283b206', 'ROe3HXFOCJntN2', 'n6zXLMoUlXggSK', 'ReDp7bf2Y7z0Cg', 'm3qKTfQtrj4Wwe', 'IFIUTiXk5o475N', 'uKuFHXmBVKofnG', 'iMQWcd52IVKymw', 'psVk4VH1iTa4uD', 'oG86LioLpw5B45', 'M5iS694oSRCBMY', '9nblxPWUklwzZw', 'aBp3qcgbCtMRNd', 'pyqiC8DqDzKGhP', '2EDydB5J6PL5FR', 'av5d2D8Ly9dpod', 'p9ylylubEtUX3U', '6O0i4hIHwv7bkP', 'AGMxSK9h1bbxMx', 'EQiH5zWD5hYxGx', 'aN3NeDE9DerRAU', '8bZ6Xh4N5myYtu', 'SkdHxLfIytCcuv', '1yg1aQTU3W69g1', '6jG5Ls3jVqwvDb', 'cv7eS8fl5B5LKT', 'EwdRhXmpYnBNIH', 'bHJDQm1y3CdT71', '7WcnYnLmxmzUiZ', 'cLsqA4HUINCWyM', '4F3be19ULlTWE7', 'VV9qZm56hp1vLS', 'cachP7YABuLmlZ', 'eFP2O4FzjBCjnv', 'VMwMXxCcAOUkOU', 'ahldbNfbmaz6f5', 'E6jMYov7gDbBkF', 'BXq4s9CF9O6xLY', 'bXATcDFvQiWAz9', '0TWfQ3p0NkLVVh', 'w0X2MBXCMwQWtL', 'V7Pp6I45ym2urq', 'A9XiKJvRJED3R9', 'WfPxCnROEcPHCK', 'Z2nJlEDNE6j0QC', '8n6yidmzZyf2B5', '5c3C1iFrxPtOZ6', 'ehaHG4yWtpLDIm', '01AQOFMDLTETDx', 'WTBykBERhXIAKQ', 'gC4lftjNK2j2Pi', 'CU2zz6ctWSSL0N', '5DdkQ35d6mlRmi', 'NH27JeMCatChSJ', 'Ou4KxVwEuU2Hli', '0cC7GPRFAIvP5i', 'UUrugKjZi5cKi6', '1eTVjMYXkOBq6b', 'dUAekwR1ljMZI0', 'mhoCamND1z5iDI', 'ZYhwzM23rFE73H', 'uvOoyHYbTvum5s', '1JQ8tLwWMnJtSO', '8Xc4I9sDrIUh7r', 'qdPmKvqLXaIz8Z', '847S4RRkprf1Ua', 'O9TwfqDYdaGr61', 'HpoxTEIaFNRf9B', '6DwshX5XDTpa6A', 'p8vfL3TYpUy4CC', 'COUaIp57kttcsD', 'kCNOwm8bfowLiK', 'ksPNeMndi5JULg', 'BD8amxQXSVO23W', '2s08sOZGPYt3a1', '0YjmvQadoCYXFu', '2qLHq7WLYuGcAE', 'aY9FyqQKI3p7xl', '9vXvg3JS3cSaCL', 'KHEXQUSO2FW8lV', 'bwgYxJcUCdOsZY', '3hOqeoHHE7oe5e', '8MZzfKW13mjTKc', 'KLjaqbTFw45Xuv', '7I9bVALgifkCnn', 'JhAIqq04R92r9f', 'LhjIcQ9tgYJNNR', 'G6xIOjpDUUvw9t', 'F5hgQTKva1uLoG', 'dnaaLoMArqPV4k', 'dsxIMPAdSSwT0n', 'dxRqhPk3HIxrM4', 'LXFL0ErRAsYB0u', 'J92tVPIyrSUSsv', 'HG4Ti53aafpkP7', 'S8rgPIRmIi3OlV', 'nb2tnbMgYy2nuR', 'iYy9SwaytWbGbS', 'eCusCKEVPULHBw', 'NBaAnpXsN67cAv', 'rYytHeXZY1zAo1', 'Nmvl6g6p1SkLiI', 'dPNzr5Iw2AjKtI', 'hSCLBNq4kziTpZ', 'LW9Gdr7VrA3Y5G', 'AvSVVlFPTaSaxb', 'EIcg9xr8J7UP8R', 'gSNx3FQy3IPBB3', 'gl5JWPV8RbUnJc', 'luRNTGvWfmVJlu', 'fmCi8P8gJO4XNP', 'bxpUypRLyU7kK4', 'BVH8jmUsoXxEWY', 'ocHILgJEBwe3Fj', 'ODw2HRgMV4AE05', '4NrFH9urQTa150', 'c4mBm9m2tWnSKh', 'CNtIXaJI7tTdjW', 'hmwBEvR40ljLKR', 'CilScWOB3ShJZ1', 'KF5Za23vo4J04c', 'delbyb1J97oqWr', '1TYXs1QBtXUbiT', 'JGxjYyAxacBkVZ', 'IvVpe1HpolCVug', '7Fu3FqV9jH1SJT', 'JVyaNYM3I6bZZn', 'cju3JFuqtEBgec', 'oMrTKZLQ2ynV3Z', '4bFbfTBMnBflT3', '3WXcA4zGMPQjBK']

VAL_IMAGES_ALL = [
    "sideboard", "cotton", "boxes", "dino",
    "antinous", "greek", "dishes", "tower",
    "backgammon", "pyramids", "stripes", "dots",
    "rosemary", "boardgames", "museum", "pillows", "tomb", "vinyl",
    "kitchen", "medieval2", "pens", "platonic", "table", "town",
    # 'buddha', 'buddha2', 'monasRoom', 'papillon', 'stillLife'
]

TEST_IMAGES = [
    "sideboard", "cotton", "boxes", "dino",
    # "antinous","greek","dishes","tower",
    "backgammon", "pyramids", "stripes", "dots",
    "bedroom", "bicycle", "herbs", "origami"
    # "rosemary","boardgames","museum","pillows","tomb","vinyl",
    # "kitchen","medieval2","pens","platonic","table","town",
    # 'buddha', 'buddha2', 'monasRoom', 'papillon', 'stillLife'
]
RMSE_MAE_BPR_IMAGES = ["kitchen", "pillows", "tomb", "vinyl"]
OLD_TEST_IMAGES = [
    "maria", "statue", "buddha2", "stillLife", "cube", "papillon", "couple", "buddha", "monasRoom", "pyramide"
]


def get_scores(img, nb):
    """
    this funciton  gets the scores
    :param img: numpy shape[h,w]
    :param nb: sequence of the val img
    :return:score
    """
    ERROR_RANGE = 0.07
    bp007 = BadPix(0.07)
    evaluator = Evaluator()
    category = my_misc.infer_scene_category(VAL_IMAGES[nb])
    # Backgammon, Stripes will set gt_scale to 10 when compute_scores(), so leave it 1.0 here
    EVAL_ROOT = "/root/Algorithm/UnLFdisp/evaluation_toolkit/data/full_data"
    sceneEval = my_misc.get_scene(VAL_IMAGES[nb], category, data_path=EVAL_ROOT)
    pre_err = eval_tools.compute_scores(sceneEval, [bp007], img)[bp007.get_id()]['value']
    print("-----------------scene {}---------------").format(VAL_IMAGES[nb])
    print("Value({}) accuracy:right rate {:.3f} error rate {:.3f}".format(ERROR_RANGE, 100 - pre_err, pre_err))
    return pre_err


def error_params_for_plt(err, title=""):
    # params refer to p_utils.show_img(img, title="", norm=None, show_axes_ticks=True, with_colorbar=False)
    err_norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    return (err, title, err_norm, False, True)


def get_scores_file(img, nb, myfile):
    """
    this funciton  gets the scores
    :param img: numpy shape[h,w]
    :param nb: sequence of the val img
    :return:score
    """
    if myfile is None:
        print("please send into the file")
        return
    ERROR_RANGE = 0.07
    bp007 = BadPix(0.07)  # 0.5
    category = my_misc.infer_scene_category(VAL_IMAGES[nb])
    # Backgammon, Stripes will set gt_scale to 10 when compute_scores(), so leave it 1.0 here
    EVAL_ROOT = "/root/Algorithm/UnLFdisp/evaluation_toolkit/data/full_data"
    sceneEval = my_misc.get_scene(VAL_IMAGES[nb], category, data_path=EVAL_ROOT)
    pre_err_new = eval_tools.compute_errorimage(sceneEval, [bp007], img, True)[1][0]

    # fig = plt.figure(figsize=(24, 6))
    # # plt.figure(figsize=(6, 6.5))
    # ax = fig.add_subplot(131)
    # ax.set_title("result")
    # ax.imshow(img)#gt
    # ax = fig.add_subplot(132)
    # ax.set_title("error img")
    # ax.imshow(pre_err_new)
    # ax = fig.add_subplot(133)#result
    # ax.set_title("reder is error img")
    # im = ax.imshow(np.subtract(1.,pre_err_new), cmap=plt.cm.autumn)
    # plt.colorbar(im)
    # figname = VAL_IMAGES[nb]+".png"
    # fig.savefig('figname')

    pre_err = eval_tools.compute_scores(sceneEval, [bp007], img)[bp007.get_id()]['value']
    # disp_gt = np.load("{}/{}.npy".format(f_dir, f_name_split))
    # disp_gt = disp_gt.astype(np.float32)
    # disp_err_pre, value_acc = evaluator.error_acc(img, disp_gt, eval_mask=None)
    print("-----------------scene {}---------------").format(VAL_IMAGES[nb])
    print("Value({}) accuracy:right rate {:.3f} error rate {:.3f}".format(ERROR_RANGE, 100 - pre_err, pre_err))
    myfile.write("-----------------scene {}---------------".format(VAL_IMAGES[nb] + '\n'))
    myfile.write(
        "Value({}) accuracy:right rate {:.3f} error rate {:.3f}".format(ERROR_RANGE, 100 - pre_err, pre_err) + '\n')
    myfile.flush()
    return pre_err_new, pre_err


def get_scores_file_by_name_clear(img, scene_name, myfile):
    """
    this funciton  gets the scores
    :param img: numpy shape[h,w]
    :param nb: sequence of the val img
    :return:score
    """
    if myfile is None:
        print("please send into the file")
        return
    ERROR_RANGE = 0.07
    bp007 = BadPix(0.07)  # 0.5
    category = my_misc.infer_scene_category(scene_name)
    # Backgammon, Stripes will set gt_scale to 10 when compute_scores(), so leave it 1.0 here
    EVAL_ROOT = "/root/Algorithm/UnLFdisp/evaluation_toolkit/data/full_data"
    sceneEval = my_misc.get_scene(scene_name, category, data_path=EVAL_ROOT)
    pre_err_new = eval_tools.compute_errorimage(sceneEval, [bp007], img, True)[1][0]

    pre_err = eval_tools.compute_scores(sceneEval, [bp007], img)[bp007.get_id()]['value']
    # print ("-----------------scene {}---------------").format(scene_name)
    # print ("Value({}) accuracy:right rate {:.3f} error rate {:.3f}".format(ERROR_RANGE, 100 - pre_err, pre_err))
    myfile.write("-----------------scene {}---------------".format(scene_name + '\n'))
    myfile.write(
        "Value({}) accuracy:right rate {:.3f} error rate {:.3f}".format(ERROR_RANGE, 100 - pre_err, pre_err) + '\n')
    myfile.flush()
    return pre_err_new, pre_err


def get_scores_file_by_name(img, scene_name, myfile):
    """
    this funciton  gets the scores
    :param img: numpy shape[h,w]
    :param nb: sequence of the val img
    :return:score
    """
    if myfile is None:
        print("please send into the file")
        return
    ERROR_RANGE = 0.07
    bp007 = BadPix(0.07)  # 0.5
    category = my_misc.infer_scene_category(scene_name)
    # Backgammon, Stripes will set gt_scale to 10 when compute_scores(), so leave it 1.0 here
    EVAL_ROOT = "/root/Algorithm/UnLFdisp/evaluation_toolkit/data/full_data"
    sceneEval = my_misc.get_scene(scene_name, category, data_path=EVAL_ROOT)
    pre_err_new = eval_tools.compute_errorimage(sceneEval, [bp007], img, True)[1][0]

    pre_err = eval_tools.compute_scores(sceneEval, [bp007], img)[bp007.get_id()]['value']
    print("-----------------scene {}---------------".format(scene_name))
    print("Value({}) accuracy:right rate {:.3f} error rate {:.3f}".format(ERROR_RANGE, 100 - pre_err, pre_err))
    myfile.write("-----------------scene {}---------------".format(scene_name + '\n'))
    myfile.write(
        "Value({}) accuracy:right rate {:.3f} error rate {:.3f}".format(ERROR_RANGE, 100 - pre_err, pre_err) + '\n')
    myfile.flush()
    return pre_err_new, pre_err


def save_erroplt_by_name(groundtruth, myresult, error_img, output_directory, scene_name, is_pp):
    """
    save the plt into the output_directory
    :param myresult:
    :param error_img:
    :param output_directory:
    :return:
    """
    bp007 = BadPix(0.07)
    fig = plt.figure(figsize=(24, 6))
    # plt.figure(figsize=(6, 6.5))
    ax = fig.add_subplot(141)
    ax.set_title("gt")
    ax.imshow(groundtruth)
    ax = fig.add_subplot(142)
    ax.set_title("result")
    ax.imshow(myresult)
    ax = fig.add_subplot(143)
    ax.set_title("error img")
    ax.imshow(error_img, **setting.metric_args(bp007))

    if is_pp == 0:
        fignamepath = os.path.join(output_directory, scene_name + "disp1_fig" + ".png")
    if is_pp == 1:
        fignamepath = os.path.join(output_directory, scene_name + "disp1_pp_fig.png")
    if is_pp == 2:
        fignamepath = os.path.join(output_directory, scene_name + "disp0_fig.png")
    if is_pp == 3:
        fignamepath = os.path.join(output_directory, scene_name + "disp0_pp_fig.png")
    fig.savefig(fignamepath)
    plt.close()
    return 0


def get_scores_file_RMSE_MAE_BPR(img, nb, myfile):
    """
    this funciton  gets the scores
    :param img: numpy shape[h,w]
    :param nb: sequence of the val img
    :return:score
    """
    if myfile is None:
        print("please send into the file")
        return
    ERROR_RANGE = 0.07
    bp007 = BadPix(0.07)  # 0.5
    category = my_misc.infer_scene_category(RMSE_MAE_BPR_IMAGES[nb])
    # Backgammon, Stripes will set gt_scale to 10 when compute_scores(), so leave it 1.0 here
    EVAL_ROOT = "/root/Algorithm/UnLFdisp/evaluation_toolkit/data/full_data"
    sceneEval = my_misc.get_scene(RMSE_MAE_BPR_IMAGES[nb], category, data_path=EVAL_ROOT)
    pre_err_new = eval_tools.compute_scores(sceneEval, [bp007], img, True)[1][0]

    # fig = plt.figure(figsize=(24, 6))
    # # plt.figure(figsize=(6, 6.5))
    # ax = fig.add_subplot(131)
    # ax.set_title("result")
    # ax.imshow(img)#gt
    # ax = fig.add_subplot(132)
    # ax.set_title("error img")
    # ax.imshow(pre_err_new)
    # ax = fig.add_subplot(133)#result
    # ax.set_title("reder is error img")
    # im = ax.imshow(np.subtract(1.,pre_err_new), cmap=plt.cm.autumn)
    # plt.colorbar(im)
    # figname = VAL_IMAGES[nb]+".png"
    # fig.savefig('figname')

    pre_err = eval_tools.compute_scores(sceneEval, [bp007], img)[bp007.get_id()]['value']
    # disp_gt = np.load("{}/{}.npy".format(f_dir, f_name_split))
    # disp_gt = disp_gt.astype(np.float32)
    # disp_err_pre, value_acc = evaluator.error_acc(img, disp_gt, eval_mask=None)
    print("-----------------scene {}---------------").format(RMSE_MAE_BPR_IMAGES[nb])
    print("Value({}) accuracy:right rate {:.3f} error rate {:.3f}".format(ERROR_RANGE, 100 - pre_err, pre_err))
    myfile.write("-----------------scene {}---------------".format(RMSE_MAE_BPR_IMAGES[nb] + '\n'))
    myfile.write(
        "Value({}) accuracy:right rate {:.3f} error rate {:.3f}".format(ERROR_RANGE, 100 - pre_err, pre_err) + '\n')
    myfile.flush()
    return pre_err_new, pre_err


def get_scores_file_all(img, nb, myfile):
    """
    this funciton  gets the scores
    :param img: numpy shape[h,w]
    :param nb: sequence of the val img
    :return:score
    """
    if myfile is None:
        print("please send into the file")
        return
    ERROR_RANGE = 0.07
    bp007 = BadPix(0.07)  # 0.5
    category = my_misc.infer_scene_category(VAL_IMAGES_ALL[nb])
    # Backgammon, Stripes will set gt_scale to 10 when compute_scores(), so leave it 1.0 here
    EVAL_ROOT = "/root/Algorithm/UnLFdisp/evaluation_toolkit/data/full_data"
    sceneEval = my_misc.get_scene(VAL_IMAGES_ALL[nb], category, data_path=EVAL_ROOT)
    pre_err_new = eval_tools.compute_scores(sceneEval, [bp007], img, True)[1][0]

    # fig = plt.figure(figsize=(24, 6))
    # # plt.figure(figsize=(6, 6.5))
    # ax = fig.add_subplot(131)
    # ax.set_title("result")
    # ax.imshow(img)#gt
    # ax = fig.add_subplot(132)
    # ax.set_title("error img")
    # ax.imshow(pre_err_new)
    # ax = fig.add_subplot(133)#result
    # ax.set_title("reder is error img")
    # im = ax.imshow(np.subtract(1.,pre_err_new), cmap=plt.cm.autumn)
    # plt.colorbar(im)
    # figname = VAL_IMAGES[nb]+".png"
    # fig.savefig('figname')

    pre_err = eval_tools.compute_scores(sceneEval, [bp007], img)[bp007.get_id()]['value']
    # disp_gt = np.load("{}/{}.npy".format(f_dir, f_name_split))
    # disp_gt = disp_gt.astype(np.float32)
    # disp_err_pre, value_acc = evaluator.error_acc(img, disp_gt, eval_mask=None)
    print("-----------------scene {}---------------").format(VAL_IMAGES_ALL[nb])
    print("Value({}) accuracy:right rate {:.3f} error rate {:.3f}".format(ERROR_RANGE, 100 - pre_err, pre_err))
    myfile.write("-----------------scene {}---------------".format(VAL_IMAGES_ALL[nb] + '\n'))
    myfile.write(
        "Value({}) accuracy:right rate {:.3f} error rate {:.3f}".format(ERROR_RANGE, 100 - pre_err, pre_err) + '\n')
    myfile.flush()
    return pre_err_new, pre_err


def save_erroplt_all(groundtruth, myresult, error_img, output_directory, image_number, is_pp):
    """
    save the plt into the output_directory
    :param myresult:
    :param error_img:
    :param output_directory:
    :return:
    """
    fig = plt.figure(figsize=(24, 6))
    # plt.figure(figsize=(6, 6.5))
    ax = fig.add_subplot(141)
    ax.set_title("gt")
    ax.imshow(groundtruth)
    ax = fig.add_subplot(142)
    ax.set_title("result")
    ax.imshow(myresult)
    ax = fig.add_subplot(143)
    ax.set_title("error img")
    ax.imshow(error_img)
    # ax = fig.add_subplot(144)  # result
    # ax.set_title("reder is error img")
    # im = ax.imshow(np.subtract(1., error_img), cmap=plt.cm.autumn)
    # plt.colorbar(im)
    if is_pp:
        # fignamepath = os.path.join(output_directory,VAL_IMAGES[image_number]+"_pp_fig" + ".png")
        fignamepath = os.path.join(output_directory, VAL_IMAGES_ALL[image_number] + "_pp_fig" + ".png")
    else:
        # fignamepath = os.path.join(output_directory, VAL_IMAGES[image_number] + "fig.png")
        fignamepath = os.path.join(output_directory, VAL_IMAGES_ALL[image_number] + "fig.png")
    fig.savefig(fignamepath)
    plt.close()
    return 0


def save_erroplt(groundtruth, myresult, error_img, output_directory, image_number, is_pp):
    """
    save the plt into the output_directory
    :param myresult:
    :param error_img:
    :param output_directory:
    :return:
    """
    bp007 = BadPix(0.07)
    fig = plt.figure(figsize=(24, 6))
    # plt.figure(figsize=(6, 6.5))
    ax = fig.add_subplot(141)
    ax.set_title("gt")
    ax.imshow(groundtruth)
    ax = fig.add_subplot(142)
    ax.set_title("result")
    ax.imshow(myresult)
    ax = fig.add_subplot(143)
    ax.set_title("error img")
    ax.imshow(error_img, **setting.metric_args(bp007))
    # ax = fig.add_subplot(144)  # result
    # ax.set_title("reder is error img")
    # im = ax.imshow(np.subtract(1., error_img), cmap=plt.cm.autumn)
    # plt.colorbar(im)
    if is_pp:
        fignamepath = os.path.join(output_directory, VAL_IMAGES[image_number] + "_pp_fig" + ".png")
    else:
        fignamepath = os.path.join(output_directory, VAL_IMAGES[image_number] + "fig.png")
    fig.savefig(fignamepath)
    plt.close()
    return 0


def write_pfm(data, fpath, scale=1, file_identifier=b'Pf', dtype="float32"):
    # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html

    data = np.flipud(data)
    height, width = np.shape(data)[:2]
    values = np.ndarray.flatten(np.asarray(data, dtype=dtype))
    endianess = data.dtype.byteorder

    if endianess == '<' or (endianess == '=' and sys.byteorder == 'little'):
        scale *= -1

    with open(fpath, 'wb') as file:
        file.write((file_identifier))
        file.write(('\n%d %d\n' % (width, height)).encode())
        file.write(('%d\n' % scale).encode())

        file.write(values)


# def read_pfm(fpath, expected_identifier="Pf"):
#     # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html
#
#     def _get_next_line(f):
#         next_line = f.readline().decode('utf-8').rstrip()
#         # ignore comments
#         while next_line.startswith('#'):
#             next_line = f.readline().rstrip()
#         return next_line
#
#     with open(fpath, 'rb') as f:
#         #  header
#         identifier = _get_next_line(f)
#         if identifier != expected_identifier:
#             raise Exception('Unknown identifier. Expected: "%s", got: "%s".' % (expected_identifier, identifier))
#
#         try:
#             line_dimensions = _get_next_line(f)
#             dimensions = line_dimensions.split(' ')
#             width = int(dimensions[0].strip())
#             height = int(dimensions[1].strip())
#         except:
#             raise Exception('Could not parse dimensions: "%s". '
#                             'Expected "width height", e.g. "512 512".' % line_dimensions)
#
#         try:
#             line_scale = _get_next_line(f)
#             scale = float(line_scale)
#             assert scale != 0
#             if scale < 0:
#                 endianness = "<"
#             else:
#                 endianness = ">"
#         except:
#             raise Exception('Could not parse max value / endianess information: "%s". '
#                             'Should be a non-zero number.' % line_scale)
#
#         try:
#             data = np.fromfile(f, "%sf" % endianness)
#             data = np.reshape(data, (height, width))
#             data = np.flipud(data)
#             with np.errstate(invalid="ignore"):
#                 data *= abs(scale)
#         except:
#             raise Exception('Invalid binary values. Could not create %dx%d array from input.' % (height, width))
#
#         return data


def save_singleimg(data, fpath, fname):
    # fig = plt.figure(figsize=(5.12, 5.12))
    # plt.axis('off')
    # ax = fig.add_subplot(111)
    #
    # ax.imshow(data)
    # fignamepath = os.path.join(fpath, fname + "fig.png")
    # fig.savefig(fignamepath,bbox_inches='tight')
    # plt.close()

    fig, ax = plt.subplots()
    im = data
    ax.imshow(im, aspect='equal')
    plt.axis('off')
    height, width, channel = np.shape(im)
    fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    fignamepath = os.path.join(fpath, fname + ".png")
    plt.savefig(fignamepath, dpi=300)
    return


def save_singledisp(data, fpath, fname):
    # fig = plt.figure(figsize=(5.12, 5.12))
    # plt.axis('off')
    # ax = fig.add_subplot(111)
    #
    # ax.imshow(data)
    # fignamepath = os.path.join(fpath, fname + "fig.png")
    # fig.savefig(fignamepath,bbox_inches='tight')
    # plt.close()

    fig, ax = plt.subplots()
    im = data

    # bp007 = BadPix(0.07)
    # ax.imshow(im, aspect='equal',**setting.metric_args(bp007))
    ax.imshow(im, aspect='equal')
    plt.axis('off')
    height, width = np.shape(im)
    fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    fignamepath = os.path.join(fpath, fname + "fig.png")
    plt.savefig(fignamepath, dpi=300)
    plt.close()
    return


def save_singledisp_error(data, fpath, fname):
    # fig = plt.figure(figsize=(5.12, 5.12))
    # plt.axis('off')
    # ax = fig.add_subplot(111)
    #
    # ax.imshow(data)
    # fignamepath = os.path.join(fpath, fname + "fig.png")
    # fig.savefig(fignamepath,bbox_inches='tight')
    # plt.close()

    fig, ax = plt.subplots()
    # im = data
    im = data[15:-15, 15:-15]

    bp007 = BadPix(0.07)
    ax.imshow(im, aspect='equal', **setting.metric_args(bp007))
    # ax.imshow(im, aspect='equal')
    plt.axis('off')
    height, width = np.shape(im)
    fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    fignamepath = os.path.join(fpath, fname + "fig.png")
    plt.savefig(fignamepath, dpi=300)
    plt.close()
    return


def cal_RMSE(disp, gt):
    rmse_pow = np.power(np.subtract(disp, gt), 2)
    rmse_result = np.sqrt(np.mean(rmse_pow))

    mae_result = np.mean(np.abs(np.subtract(disp, gt)))

    bpr_abs = np.abs(np.subtract(disp, gt))
    bpr = np.where(bpr_abs > 0.2, 1, 0)
    bpr_result = np.mean(bpr)

    return rmse_result, mae_result, bpr_result
