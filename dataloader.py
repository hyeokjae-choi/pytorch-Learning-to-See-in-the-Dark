import os
import os.path as osp
import glob

from skimage.io import imread
import numpy as np
import cv2


def rgb2input(rgb, ratio=1):
    img = rgb.astype(np.float32) / 255

    img_r = img[:, :, 0]
    img_g = img[:, :, 1] * 2
    img_b = img[:, :, 2]

    inp = np.stack([img_r, img_g, img_b, img_g], axis=-1)
    inp = np.expand_dims(inp, axis=0) * ratio
    inp = np.minimum(inp, 1.0)
    return inp


class ClaDataset:
    def __init__(self, data_dir):
        self.samples = glob.glob(osp.join(data_dir, "*", "*.mp4"))

    def __getitem__(self, index):
        fpath = self.samples[index]
        fname = osp.basename(fpath)
        label = fpath.split(os.sep)[-2]

        img_list = []
        # read video
        cap = cv2.VideoCapture(fpath)
        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                break
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # pad image to multiples of 32
            size_gap = [int((32 - (i % 32)) / 2) for i in img.shape[:2]]
            img = np.pad(img, ((size_gap[0], size_gap[0]), (size_gap[1], size_gap[1]), (0, 0)))

            # convert to "learning to dark" style
            img = rgb2input(img)
            img_list.append(img[0])

        img_list = np.stack(img_list, axis=0)
        return img_list, fname, label, size_gap

    def __len__(self):
        return len(self.samples)
