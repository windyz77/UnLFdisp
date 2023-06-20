import os

# filename = "/home/jethong/PycharmProjects/multi_input/4dlf_7x7star_add_train.txt"
save_txt = "/home/jethong/PycharmProjects/multi_input/real_scene/test.txt"
# savetest_txt = "/home/jethong/PycharmProjects/multi_input/unos_iter/iter_3_test.txt"
frontname = "full_data/realword_dataset/{}/{}/input_Cam{}.png"

filename = "/home/jethong/data/full_data/realword_dataset"

# t = [(5, 5), (2, 2), (3, 3), (4, 4), (6, 6), (7, 7), (8, 8),
#      (2, 8), (3, 7), (4, 6), (6, 4), (7, 3), (8, 2),
#      (5, 2), (5, 3), (5, 4), (5, 6), (5, 7), (5, 8),
#      (2, 5), (3, 5), (4, 5), (6, 5), (7, 5), (8, 5)]
# d = [-10, -9, -8, -1, 0, 1, 8, 9, 10]
d = [0]
t = [40, 10, 20, 30, 50, 60, 70, 16, 24, 32, 48, 56, 64, 37, 38, 39, 41, 42, 43, 13, 22, 31, 49, 58, 67]
for dir in os.listdir(filename):
    for scene in os.listdir(filename + "/" + dir):
        for i in d:
            total = ""
            for j in t:
                x = i + j
                total += frontname.format(dir, scene, str(x).zfill(3)) + " "
            total += "\n"
            with open(save_txt, "a+") as f1:
                f1.write(total)
