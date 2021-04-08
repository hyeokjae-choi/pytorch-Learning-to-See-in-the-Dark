import os
import os.path as osp
import glob
import multiprocessing
from functools import partial
import warnings

import numpy as np
import cv2
from skimage.io import imsave

warnings.simplefilter("ignore", UserWarning)


def read_video(video_fpath):
    frame_list = []
    cap = cv2.VideoCapture(video_fpath)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_list.append(frame)
    return frame_list


def save_video(frame_list, save_fpath, fourcc=cv2.VideoWriter_fourcc(*"XVID"), fps=30):
    size = frame_list[0].shape[:2]
    out = cv2.VideoWriter(save_fpath, fourcc, fps, size[::-1])
    for frame in frame_list:
        out.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    out.release()


def _video2images(save_dir, video_fpath):
    label = video_fpath.split(os.sep)[-2]
    if not osp.isdir(osp.join(save_dir, label)):
        os.makedirs(osp.join(save_dir, label))

    image_list = read_video(video_fpath)
    for i, image in enumerate(image_list):
        save_fpath = osp.join(save_dir, label, f"{osp.splitext(osp.basename(video_fpath))[0]}_{i:03d}.png")
        imsave(save_fpath, image)


def video2images():
    div = "Validation"

    data_dir = r"F:\sualab\2021\1Q\workshop\data\UG2-2021-Track2.1\video"
    save_dir = r"F:\sualab\2021\1Q\workshop\data\UG2-2021-Track2.1\image"

    video_list = glob.glob(osp.join(data_dir, div, "*", "*.mp4"))

    num_cores = 8  # multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_cores) as pool:
        func = partial(_video2images, osp.join(save_dir, div))
        pool.map(func, video_list)


def _video_smoothing(le_data_dir, div, save_dir, raw_video_fpath, smoothing_method="avg_diff"):
    fname = osp.splitext(osp.basename(raw_video_fpath))[0]
    label = raw_video_fpath.split(os.sep)[-2]

    raw_frame_list = read_video(raw_video_fpath)
    le_frame_list = read_video(osp.join(le_data_dir, div, label, f"{fname}.avi"))

    if smoothing_method == "avg_diff":
        weight = [1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5]
        # get difference frame between raw and le
        diff_frame_list = []
        for raw, le in zip(raw_frame_list, le_frame_list):
            raw = raw.astype(np.float32)
            le = le.astype(np.float32)

            diff_frame = raw - le
            diff_frame_list.append(diff_frame)

        # get smoothed le
        smooth_frame_list = []
        for i, raw in enumerate(raw_frame_list):
            smooth = []
            for j, w in enumerate(weight):
                # handling overflow
                idx = i + j - int(len(weight) / 2)
                idx = np.clip(idx, 0, len(diff_frame_list) - 1)
                smooth.append(w * diff_frame_list[idx])
            smooth = np.stack(smooth)
            smooth = np.sum(smooth, axis=0)

            le = raw.astype(np.float32) - smooth
            le = np.clip(le, 0, 255)
            le = le.astype(np.uint8)

            smooth_frame_list.append(le)

    elif smoothing_method == "denoising":
        num_frame = 5

        smooth_frame_list = []
        for i, raw in enumerate(le_frame_list):
            smooth = []
            for j in range(num_frame):
                # handling overflow
                idx = i + j - int(num_frame / 2)
                idx = np.clip(idx, 0, len(le_frame_list) - 1)
                smooth.append(le_frame_list[idx])

            smooth = cv2.fastNlMeansDenoisingMulti(smooth, 2, 5, None, 4, 7, 35)
            smooth_frame_list.append(smooth)

    # save results
    if not osp.isdir(osp.join(save_dir, div, label)):
        os.makedirs(osp.join(save_dir, div, label))
    save_video(smooth_frame_list, osp.join(save_dir, div, label, f"{fname}.avi"))


def video_smoothing():
    div = "Validation"

    raw_data_dir = "/data/hyeokjae/data/UG2-2021-Track2.1/video"
    le_data_dir = "/data/hyeokjae/results/UG2-2021/light_enhancement/sid/Track2.1"
    save_dir = "/data/hyeokjae/results/UG2-2021/light_enhancement/sid/Track2.1_denoising"

    raw_video_list = glob.glob(osp.join(raw_data_dir, div, "*", "*"))

    num_cores = 1  # multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_cores) as pool:
        func = partial(_video_smoothing, le_data_dir, div, save_dir)
        pool.map(func, raw_video_list)


if __name__ == "__main__":
    # video2images()
    video_smoothing()
