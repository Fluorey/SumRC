"""把原数据集划分为train/test/val三个子集"""

import os
import shutil
import random
from tqdm import tqdm


def raw_data_split(cnn_dm=True, data_folder="cnn_dm"):
    train_ratio = 0.92
    test_ratio = 0.043
    val_ratio = 0.038

    file_list = os.listdir(data_folder)
    random.shuffle(file_list)  # 随机化文件列表的顺序

    total_files = len(file_list)
    train_files = int(train_ratio * total_files)
    test_files = int(test_ratio * total_files)

    train_set = file_list[:train_files]
    test_set = file_list[train_files:train_files + test_files]
    val_set = file_list[train_files + test_files:]

    # 创建目标文件夹
    if cnn_dm:
        train_folder = "raw_split/cnn_dm/train"
        test_folder = "raw_split/cnn_dm/test"
        val_folder = "raw_split/cnn_dm/val"
    else:
        train_folder = "raw_split/xsum/train"
        test_folder = "raw_split/xsum/test"
        val_folder = "raw_split/xsum/val"

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    # 复制文件
    for file in tqdm(train_set, desc="训练集划分", ascii=True):
        shutil.copy(os.path.join(data_folder, file), os.path.join(train_folder, file))

    for file in tqdm(test_set, desc="测试集划分", ascii=True):
        shutil.copy(os.path.join(data_folder, file), os.path.join(test_folder, file))

    for file in tqdm(val_set, desc="验证集划分", ascii=True):
        shutil.copy(os.path.join(data_folder, file), os.path.join(val_folder, file))
