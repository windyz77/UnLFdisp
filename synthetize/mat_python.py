import sys
sys.path.append(r"/root/Algorithm/UnLFdisp")
from file_io import *
import time
import cv2

t = [1, 2, 3, 5, 6, 7, 10, 11, 12, 14, 15, 16, 19, 20, 21, 23, 24, 25, 28, 29, 30, 32, 33, 34]
front_name = "/inputCam_{}_our.png"
def func(disp):
    result = []
    for b in range(1):
        disp[b].tofile("disp.ha")
        os.system('./process_v1_mt8')
        r2 = np.fromfile("occlumap.ha", dtype=np.float32).reshape(25, 512, 512, 1)
        result.append(r2)
    return np.stack(result, axis=1)
