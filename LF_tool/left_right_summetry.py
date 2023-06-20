from file_io import *
import cv2

fileright = "/home/jethong/data/full_data/mask_view_9x9/boxes/inputCam_001.png"
fileleft = "/home/jethong/data/full_data/mask_view_9x9/boxes/inputCam_009.png"

pfmleft = "/home/jethong/PycharmProjects/MaskFlwonet_LF/left_mid/2.pfm"
pfmright = "/home/jethong/PycharmProjects/MaskFlwonet_LF/mid_right/2.pfm"

png_left = cv2.imread(fileleft)[:, :, 0] / 255.0
png_right = cv2.imread(fileright)[:, :, 0] / 255.0

png_left = png_left.astype(np.bool)
png_right = png_right.astype(np.bool)

png_left_inv = (png_left == False)
png_right_inv = (png_right == False)
pfm_left = read_pfm(pfmleft)
pfm_right = read_pfm(pfmright)

pfm = (pfm_left * png_left + pfm_right * png_left_inv) / 2 + (pfm_right * png_right + pfm_left * png_right_inv) / 2
write_pfm(pfm, "boxes.pfm")
