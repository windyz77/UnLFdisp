import os

save_txt = "/home/jethong/PycharmProjects/multi_input/code_UnLFdisp_5x5/txt/train_r8_5x5.txt"
frontname = "full_data/r8_5x5/{}" + "/input_Cam{}.png"
# filename = "/home/jethong/data/full_data/scenes_1080/"
d = [0]
t = [12, 0, 0, 6, 18, 24, 0, 0, 4, 8, 16, 20, 0, 0, 10, 11, 13, 14, 0, 0, 2, 7, 17, 22, 0]
for scene in range(1, 701):
    for i in d:
        total = ""
        for j in t:
            x = i + j
            total += frontname.format("frame_" + str(scene), str(x).zfill(3)) + " "
        total += "\n"
        with open(save_txt, "a+") as f1:
            f1.write(total)
