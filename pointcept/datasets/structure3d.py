"""
Structured3D Datasets

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import glob
from collections.abc import Sequence

from .defaults import DefaultDataset
from .builder import DATASETS
import random

@DATASETS.register_module()
class Structured3DDataset(DefaultDataset):
    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = glob.glob(
                os.path.join(self.data_root, self.split, "scene_*/room_*")
            )
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(
                    os.path.join(self.data_root, split, "scene_*/room_*")
                )
        else:
            raise NotImplementedError
        # return data_list
        # xmw修改，只采用一半的数据
        data_list_half = data_list[::2]
        return data_list_half

    def get_data_name(self, idx):
        file_path = self.data_list[idx % len(self.data_list)]
        dir_path, room_name = os.path.split(file_path)
        scene_name = os.path.basename(dir_path)
        data_name = f"{scene_name}_{room_name}"
        return data_name
