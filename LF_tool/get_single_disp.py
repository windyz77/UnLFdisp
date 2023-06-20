from evalfunctions7x7 import *
from file_io import *
import os
import cv2

# dir = "/home/jethong/PycharmProjects/TIPcode_final_2020_5_5/code_for_run/flower_result/2/"
dir = "/home/fufu/data/full_data/additional/antinous/"
# dir = "/home/jethong/PycharmProjects/LFattNet-master/LFattNet_output/"


def func(pfm, file, name):
    # pfm = pfm[10:-10, 10:-10]
    pfm = (pfm - pfm.min()) / (pfm.max() - pfm.min())
    pfm = (pfm * 255.0).astype(np.uint8)
    cv2.imwrite(file + name + ".png", pfm)


#
# save_file = "./flower_1_lfatt_7x7/"
save_file = "/home/fufu/data/full_data/additional/antinous/"
# save_file = "./flower_tip_7x7/"
if not os.path.exists(save_file):
    os.mkdir(save_file)
# pfm = read_pfm("/home/jethong/PycharmProjects/multi_input/code_UnLFdisp/mask_all_data_wtc/2/input_Cam024.png full_data_lytro.pfm")
# save_singledisp(pfm, save_file, "color")

for sub in os.listdir(dir):
    if sub.endswith("pfm"):
        pfm = read_pfm(dir + sub)
        # pfm[pfm <= -2] = -2
        # func(pfm, save_file, "grey_" + sub[:-4])
        save_singledisp(pfm, save_file, sub[:-4] + "color")

#
#
# dir = "/home/jethong/PycharmProjects/multi_input/code_zhoubo_mask_v1/mix_mask_epi_loss_0_5_refine/999/"
#
# save_file = "./png"
# for sub in os.listdir(dir):
#     # if sub == "sideboard":
#         for ssub in os.listdir(dir + sub):
#             if ssub.endswith(".pfm"):
#                 pfm = read_pfm(dir + sub + "/" + ssub)
#                 func(pfm, dir + sub + "/", ssub[:-4])
