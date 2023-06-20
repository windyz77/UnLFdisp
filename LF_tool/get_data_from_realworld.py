import os

save_txt = "/home/fufu/work/code_UnLFdisp/synthetize/train0312.txt"
frontname = "synthetize/plant_0312/{}" + "/input_Cam{}.png"
filename = "/home/fufu/data/synthetize/plant_0312/"
d = [0]
# d = [-10, -9, -8, -1, 0, 1, 8, 9, 10]
t = [40, 10, 20, 30, 50, 60, 70, 16, 24, 32, 48, 56, 64, 37, 38, 39, 41, 42, 43, 13, 22, 31, 49, 58, 67]
for scene in os.listdir(filename):
    # for dir in os.listdir(filename + scene):
        for i in d:
            total = ""
            for j in t:
                x = i + j
                total += frontname.format(scene, str(x).zfill(3)) + " "
            total += "\n"
            with open(save_txt, "a+") as f1:
                f1.write(total)
