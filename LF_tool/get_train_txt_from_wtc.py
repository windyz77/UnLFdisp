import os

scene = ["train", "test", "bad", "hybrid"]

for sub_scene in scene:
    save_txt = "/home/jethong/PycharmProjects/multi_input/real_scene/train_wtc.txt"
    frontname = "full_data/wtc_dataset/" + sub_scene + "/{}/{}/input_Cam{}.png"
    filename = "/home/jethong/data/full_data/wtc_dataset/" + sub_scene

    d = [0]
    t = [24, 0, 8, 16, 32, 40, 48, 6, 12, 18, 30, 36, 42, 21, 22, 23, 25, 26, 27, 3, 10, 17, 31, 38, 45]
    for dir in os.listdir(filename):
        for scene in os.listdir(filename + "/" + dir):
            if scene.endswith("eslf"):
                for i in d:
                    total = ""
                    for j in t:
                        x = i + j
                        total += frontname.format(dir, scene, str(x).zfill(3)) + " "
                    total += "\n"
                    with open(save_txt, "a+") as f1:
                        f1.write(total)
