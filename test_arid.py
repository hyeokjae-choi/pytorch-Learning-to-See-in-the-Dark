import warnings

warnings.simplefilter("ignore", UserWarning)

import os
import os.path as osp
import glob

import numpy as np
from skimage.io import imread, imsave
import cv2
import torch
from torch.utils.data import DataLoader
import multiprocessing
from functools import partial

from model import SeeInDark
from dataloader import ClaDataset


def input2rgb(inp):
    np_inp = inp.cpu().numpy()[0].transpose((1, 2, 0))

    inp_r = np_inp[:, :, 0]
    inp_g = np_inp[:, :, 1] / 2
    inp_b = np_inp[:, :, 2]
    np_inp = np.stack([inp_r, inp_g, inp_b], axis=-1)

    return (np_inp * 255).astype("uint8")


def rgb2input(rgb, ratio=1):
    img = rgb.astype(np.float32) / 255

    img_r = img[:, :, 0]
    img_g = img[:, :, 1] * 2
    img_b = img[:, :, 2]

    inp = np.stack([img_r, img_g, img_b, img_g], axis=-1)
    inp = np.expand_dims(inp, axis=0) * ratio
    inp = np.minimum(inp, 1.0)
    inp = torch.from_numpy(inp).permute(0, 3, 1, 2).to(torch.device("cuda:0"))
    return inp


def inference(read_img, model):
    # pad image to multiples of 32
    size_gap = [int((32 - (i % 32)) / 2) for i in read_img.shape[:2]]
    img = np.pad(read_img, ((size_gap[0], size_gap[0]), (size_gap[1], size_gap[1]), (0, 0)))

    inp = rgb2input(img, ratio=1)
    # rgb = input2rgb(inp)
    # imsave(osp.join(save_dir, "converted_input.png"), rgb)

    output = model(inp)
    output = output.permute(0, 2, 3, 1).cpu().data.numpy()
    output = np.minimum(np.maximum(output, 0), 1)
    output = (255 * output[0, :, :, :]).astype("uint8")
    return output


def test_on_single():
    data_dir = "/data/hyeokjae/data/UG2-2021-Track2.1/video/Train"
    save_dir = "/data/hyeokjae/results/UG2-2021/light_enhancement/sid/UG2-2021-Track2.1/Train"
    model_fpath = r"./saved_model/checkpoint_sony_e4000.pth"

    fname = "Run_1_5_037.png"
    img_fpath = osp.join(data_dir, fname)
    read_img = imread(img_fpath)

    model = SeeInDark()
    model.load_state_dict(torch.load(model_fpath, map_location={"cuda:1": "cuda:0"}))
    model = model.to(torch.device("cuda:0"))

    output = inference(read_img, model)

    pad_w = int((output.shape[0] - read_img.shape[0]) / 2)
    pad_h = int((output.shape[1] - read_img.shape[1]) / 2)
    save_img = np.pad(read_img, ((pad_w, pad_w), (pad_h, pad_h), (0, 0)))
    save_img = np.concatenate([save_img, output], axis=1)

    imsave(osp.join(save_dir, fname), save_img)


def test_from_image():
    data_dir = "/data/hyeokjae/data/UG2-2021-Track2.1/video/Train"
    save_dir = "/data/hyeokjae/results/UG2-2021/light_enhancement/sid/UG2-2021-Track2.1/Train"
    model_fpath = r"./saved_model/checkpoint_sony_e4000.pth"

    model = SeeInDark()
    model.load_state_dict(torch.load(model_fpath, map_location={"cuda:1": "cuda:0"}))
    model = model.to(torch.device("cuda:0"))

    for label in os.listdir(data_dir):
        if not osp.isdir(osp.join(save_dir, label)):
            os.makedirs(osp.join(save_dir, label))

        img_fpath_list = glob.glob(osp.join(data_dir, label, "*.png"))
        for img_fpath in img_fpath_list:
            if osp.isfile(osp.join(save_dir, label, osp.basename(img_fpath))):
                continue

            read_img = imread(img_fpath)

            output = inference(read_img, model)

            # pad_w = int((output.shape[0] - read_img.shape[0]) / 2)
            # pad_h = int((output.shape[1] - read_img.shape[1]) / 2)
            # save_img = np.pad(read_img, ((pad_w, pad_w), (pad_h, pad_h), (0, 0)))
            # save_img = np.concatenate([save_img, output], axis=1)

            save_img = output
            imsave(osp.join(save_dir, label, osp.basename(img_fpath)), save_img)


def mp_save_images(data):
    image, fpath = data
    imsave(fpath, image)


def test_from_video():
    data_dir = "/data/hyeokjae/data/UG2-2021-Track2.1/video/Train"
    save_dir = "/data/hyeokjae/results/UG2-2021/light_enhancement/sid/UG2-2021-Track2.1/Train"
    model_fpath = r"./saved_model/checkpoint_sony_e4000.pth"

    model = SeeInDark()
    model.load_state_dict(torch.load(model_fpath, map_location={"cuda:1": "cuda:0"}))
    model = model.to(torch.device("cuda:0"))

    dataset = ClaDataset(data_dir)
    loader = DataLoader(dataset, num_workers=0)
    batch_size = 8

    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    for video_x, fname, label, pad_size in loader:
        video_x = video_x[0].permute(0, 3, 1, 2).to(torch.device("cuda:0"))

        results = []
        if video_x.shape[0] % batch_size == 0:
            num_inference = video_x.shape[0] // batch_size
        else:
            num_inference = video_x.shape[0] // batch_size + 1
        for i in range(num_inference):
            tmp_i = (i + 1) * batch_size
            tmp_i = tmp_i if tmp_i < video_x.shape[0] else video_x.shape[0]
            batch_x = video_x[i * batch_size : tmp_i]

            out = model(batch_x)
            out = out.permute(0, 2, 3, 1).cpu().data.numpy()
            out = np.minimum(np.maximum(out, 0), 1)
            out = (255 * out[:, :, :, :]).astype("uint8")
            results.extend(list(out))

        if len(results) != video_x.shape[0]:
            raise ValueError

        # resize and un-pad to original size
        save_frames = []
        pad_h, pad_w = pad_size
        for frame in results:
            h, w, _ = frame.shape
            frame = cv2.resize(frame, (int(w / 2), int(h / 2)))
            frame = frame[pad_h:-pad_h, pad_w:-pad_w]
            size = frame.shape[:2]
            save_frames.append(frame)

        if not osp.isdir(osp.join(save_dir, label[0])):
            os.makedirs(osp.join(save_dir, label[0]))

        # save video
        save_video = cv2.VideoWriter(
            osp.join(save_dir, label[0], f"{osp.splitext(fname[0])[0]}.avi"), fourcc, fps, size[::-1]
        )
        for frame in save_frames:
            save_video.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        save_video.release()

        # save image
        # for i, frame in enumerate(save_frames):
        #     imsave(osp.join(save_dir, label[0], f"{osp.splitext(fname[0])[0]}_{i:03d}.png"), frame)

        # save image with multi-pocess
        # mp_list = []
        # for i, frame in enumerate(save_frames):
        #     save_fpath = osp.join(save_dir, label[0], f"{osp.splitext(fname[0])[0]}_{i:03d}.png")
        #     mp_list.append([frame, save_fpath])
        # num_cores = 8  # multiprocessing.cpu_count()
        # with multiprocessing.Pool(processes=num_cores) as pool:
        #     func = partial(mp_save_images)
        #     pool.map(func, mp_list)


if __name__ == "__main__":
    # test_on_single()
    # test_from_image()
    test_from_video()
