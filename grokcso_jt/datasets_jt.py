import os
import re
from typing import Tuple

import numpy as np
from PIL import Image

import jittor as jt

from tools.gen_annotation import read_bounding_boxes_from_xml


def xml_2_matrix_single_jt(xml_file: str) -> np.ndarray:
    targets_gt, *_ = read_bounding_boxes_from_xml(xml_file)
    A = np.zeros((33, 33), dtype=np.float32)
    for t in targets_gt:
        x, y, lightness = t[0], t[1], t[2]
        A[int(round(3 * x + 1, 0)), int(round(3 * y + 1, 0))] = lightness
    return A


def _extract_number(filename: str) -> int:
    m = re.search(r"\d+", filename)
    return int(m.group()) if m else -1


class TrainDatasetJT:
    """
    简化版训练集：
    - 不依赖 torch.utils.data.Dataset
    - 提供按序列长度切片的接口，方便构造时序 batch
    """

    def __init__(self, data_root: str, xml_root: str):
        self.data_root = data_root
        self.xml_root = xml_root
        self.image_files = sorted(os.listdir(data_root), key=_extract_number)

    def __len__(self) -> int:
        return len(self.image_files)

    def _load_one(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_root, img_name)
        file_id = img_name[len("image_"):-len(".png")]
        xml_path = os.path.join(self.xml_root, "COS" + file_id + ".xml")

        # gt x (33x33)
        batch_x = xml_2_matrix_single_jt(xml_path).reshape(-1, 1089)  # [1,1089]

        # measurement y (11x11)
        img = Image.open(img_path)
        gt_img_11 = np.array(img).astype(np.float32).reshape(-1, 121)  # [1,121]

        return batch_x, gt_img_11

    def get_sequence(self, start: int, seq_len: int):
        end = min(start + seq_len, len(self.image_files))
        xs = []
        ys = []
        for i in range(start, end):
            bx, gy = self._load_one(i)
            xs.append(bx)
            ys.append(gy)
        x_arr = np.concatenate(xs, axis=0)  # [t,1089]
        y_arr = np.concatenate(ys, axis=0)  # [t,121]
        return jt.array(x_arr), jt.array(y_arr)


class ValDatasetJT:
    def __init__(self, data_root: str, xml_root: str):
        self.data_root = data_root
        self.xml_root = xml_root
        self.image_files = sorted(os.listdir(data_root), key=_extract_number)

    def __len__(self) -> int:
        return len(self.image_files)

    def _load_one(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_root, img_name)
        file_id = img_name[len("image_"):-len(".png")]
        xml_path = os.path.join(self.xml_root, "COS" + file_id + ".xml")

        batch_x = xml_2_matrix_single_jt(xml_path).reshape(-1, 1089)

        img = Image.open(img_path)
        gt_img_11 = np.array(img).astype(np.float32).reshape(-1, 121)

        return batch_x, gt_img_11

    def get_sequence(self, start: int, seq_len: int):
        end = min(start + seq_len, len(self.image_files))
        xs = []
        ys = []
        for i in range(start, end):
            bx, gy = self._load_one(i)
            xs.append(bx)
            ys.append(gy)
        x_arr = np.concatenate(xs, axis=0)
        y_arr = np.concatenate(ys, axis=0)
        return jt.array(x_arr), jt.array(y_arr)


