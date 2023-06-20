from file_io import *
import time
import cv2

import numpy as np


def func(disp):
    result = []
    for b in range(2):
        disp[b].tofile("disp1.ha")
        os.system('./process_v3_mt8_realscene')
        r2 = np.fromfile("occlumap1.ha", dtype=np.float32).reshape(25, 384, 512, 1)
        result.append(r2)
    return np.stack(result, axis=1)


# data = read_pfm(
#     "/media/blackops/720ee98a-d4a4-4794-bec2-6c6987ef84a4/blackops/Desktop/yg/full_data/stratified/dots/gt_disp_lowres.pfm")[
#        64:-64, :]
# data = np.stack([data, data, data, data], axis=0)
# res = func(data)
# for i in range(25):
#     cv2.imwrite(f"{i}.png", res[i,0,...] * 255.0)