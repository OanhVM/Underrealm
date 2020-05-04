from os.path import join, abspath, pardir

import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image
import shutil
import os

PATH = join(abspath(join(abspath(__file__), pardir, pardir)), "data")

raw_data_path = join(PATH, "raw_data")
raw_data_path_1 = join(raw_data_path, "20200502_1.bag")
raw_data_path_2 = join(raw_data_path, "20200502_2.bag")
raw_data_path_3 = join(raw_data_path, "20200502_3.bag")
raw_data_path_4 = join(raw_data_path, "20200502_4.bag")
raw_data_path_5 = join(raw_data_path, "20200502_5.bag")

rgb_path = join(PATH, "rgb_images")
rgb_path_1 = join(rgb_path, "1")
rgb_path_2 = join(rgb_path, "2")
rgb_path_3 = join(rgb_path, "3")
rgb_path_4 = join(rgb_path, "4")
rgb_path_5 = join(rgb_path, "5")
rgb_final = join(rgb_path, "final")

depth_path = join(PATH, "depth_images")
depth_path_1 = join(depth_path, "1")
depth_path_2 = join(depth_path, "2")
depth_path_3 = join(depth_path, "3")
depth_path_4 = join(depth_path, "4")
depth_path_5 = join(depth_path, "5")
depth_final = join(depth_path, "final")


def saving_images(frameset, number):
    color_frame = frameset.get_color_frame()

    # Align depth frame to color frame
    color = np.asanyarray(color_frame.get_data())
    align = rs.align(rs.stream.color)
    frameset = align.process(frameset)
    aligned_depth_frame = frameset.get_depth_frame()

    im = Image.fromarray(np.asanyarray(color))
    im.save(join(rgb_final, str(number) + ".png"))
    im = Image.fromarray(np.asanyarray(aligned_depth_frame.get_data()))
    im.save(join(depth_final, str(number) + ".png"))


def loading_images(rawdata):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device_from_file(rawdata)
    pipe.start(cfg)

    number = 0
    for i in range(1800):
        if i % 10 == 0:
            number += 1
            frameset = pipe.wait_for_frames()
            saving_images(frameset, number)
        else:
            pipe.wait_for_frames()

    pipe.stop()
    print("Finish loading images from .bag files ")


def change_name(r_path, d_path):
    for count, filename in enumerate(os.listdir(r_path)):
        count += 1
        print(filename)
        print(count)

        new_name = str(count) + ".png"
        old_rgb_path = join(r_path, filename)
        new_rgb_path = join(rgb_final, new_name)
        shutil.copyfile(old_rgb_path, new_rgb_path)

        old_depth_path = join(d_path, filename)
        new_depth_path = join(depth_final, new_name)
        shutil.copyfile(old_depth_path, new_depth_path)

    print("Finish changing name")


def get_depth_data(img_path):
    depth_data = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    depth_data = np.asarray(depth_data)
    depth_data = np.asarray([[j * 0.001 for j in i] for i in depth_data])
    print(depth_data[453, 205])


if __name__ == "__main__":
    pass
