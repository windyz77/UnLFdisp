import os
import cv2

filename = "/home/jethong/data/full_data/flower_1/"

file = "/home/jethong/PycharmProjects/TOOLS/LF_hci/flower_1_png/"
if not os.path.exists(file):
    os.mkdir(file)
for dir in os.listdir(filename):
    # for ssub in os.listdir(file + dir):
    img = cv2.imread(filename + dir + "/input_Cam040.png")
    cv2.imwrite(file + dir + ".png", img)
