import warnings

warnings.simplefilter("ignore", UserWarning)

import os
import os.path as osp
import glob
import multiprocessing
from functools import partial

import cv2
from skimage.io import imsave


def corruption(save_dir, video_fpath):
    cap = cv2.VideoCapture(video_fpath)

    label = video_fpath.split(os.sep)[-2]
    if not osp.isdir(osp.join(save_dir, label)):
        os.makedirs(osp.join(save_dir, label))

    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        save_fpath = osp.join(save_dir, label, f"{osp.splitext(osp.basename(video_fpath))[0]}_{i:03d}.png")
        imsave(save_fpath, frame)
        i += 1


def cvt2images():
    div = "Validation"

    data_dir = r"F:\sualab\2021\1Q\workshop\data\UG2-2021-Track2.1\video"
    save_dir = r"F:\sualab\2021\1Q\workshop\data\UG2-2021-Track2.1\image"

    video_list = glob.glob(osp.join(data_dir, div, "*", "*.mp4"))

    num_cores = 8  # multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_cores) as pool:
        func = partial(corruption, osp.join(save_dir, div))
        pool.map(func, video_list)


if __name__ == "__main__":
    cvt2images()
