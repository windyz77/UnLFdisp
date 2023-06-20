import cv2
import numpy as np
import os

filename = "/home/jethong/data/data/"
savefile = "/home/jethong/data/full_data/wtc_dataset/"


front_name = "input_Cam{}.png"
for dir in os.listdir(filename):
    if not os.path.exists(savefile + dir):
        os.mkdir(savefile + dir)
    t_file = filename + dir + "/"
    for single_dir in os.listdir(t_file):
        if not os.path.exists(savefile + dir + "/" + single_dir):
            os.mkdir(savefile + dir + "/" + single_dir)
        for single_img in os.listdir(t_file + single_dir + "/"):
            if not os.path.exists(savefile + dir + "/" + single_dir + "/" + single_img[:-4]):
                os.mkdir(savefile + dir + "/" + single_dir + "/" + single_img[:-4])
            if single_img.endswith(".png"):
                img = cv2.imread(t_file + single_dir + "/" + single_img)
                temp_file = savefile + dir + "/" + single_dir + "/" + single_img[:-4] + "/"
                cv2.imwrite("./wtc/" + dir + "_" + single_dir + "_" + single_img[:-4] + ".png", img[3::8, 3::8, :])

                # k = 0
                # for i in range(7):
                #     for j in range(7):
                #         sub_img = img[i::8, j::8, :]
                #         cv2.imwrite(temp_file+front_name.format(str(k).zfill(3)), sub_img)
                #         k += 1
