import os

save_txt = "/home/jethong/PycharmProjects/multi_input/real_scene/train_r8.txt"
frontname = "full_data/scenes_1080/{}" + "/{}/inputImg_{}.png"
filename = "/home/jethong/data/full_data/scenes_1080/"
d = [0]
t = [24, 0, 8, 16, 32, 40, 48, 6, 12, 18, 30, 36, 42, 21, 22, 23, 25, 26, 27, 3, 10, 17, 31, 38, 45]
for scene in os.listdir(filename):
    for dir in os.listdir(filename + scene):
        for i in d:
            total = ""
            for j in t:
                x = i + j
                total += frontname.format(scene, dir, str(x).zfill(3)) + " "
            total += "\n"
            with open(save_txt, "a+") as f1:
                f1.write(total)
