import os
import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from monai.transforms import Compose, RandRotate90, RandFlip, RandGaussianNoise, NormalizeIntensity


class NpyDataset(Dataset):
    def __init__(self, normal_dir, abnormal_dir, transform=None):
        self.transform = transform
        self.file_paths = []
        self.labels = []

        # 加载正常样本
        for file in os.listdir(normal_dir):
            if file.endswith('.npy'):
                self.file_paths.append(os.path.join(normal_dir, file))
                self.labels.append(0)

        # 加载异常样本
        for file in os.listdir(abnormal_dir):
            if file.endswith('.npy'):
                self.file_paths.append(os.path.join(abnormal_dir, file))
                self.labels.append(1)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx])
        data = np.expand_dims(data, axis=0)  # 添加通道维度
        data = torch.from_numpy(data).float()
        if self.transform:
            data = self.transform(data)
        label = self.labels[idx]
        return data, label


class Dataset:
    def __init__(self, config):
        transform = Compose([
            RandRotate90(),  # 随机旋转90度
            RandFlip(),  # 随机翻转
            RandGaussianNoise(),  # 高斯噪声
            NormalizeIntensity()  # 自动标准化
        ])

        train_normal_dir = os.path.join(config.processed_data_dir, 'train/HC')
        train_abnormal_dir = os.path.join(config.processed_data_dir, 'train/patient')
        train_dataset = NpyDataset(train_normal_dir, train_abnormal_dir, transform=transform)

        train_idx, val_idx = train_test_split(
            np.arange(len(train_dataset)),
            test_size=config.val_ratio,
            random_state=42,
            stratify=train_dataset.labels
        )

        self.train_dataset = Subset(train_dataset, train_idx)
        self.val_dataset = Subset(train_dataset, val_idx)

        test_normal_dir = os.path.join(config.processed_data_dir, 'test/HC')
        test_abnormal_dir = os.path.join(config.processed_data_dir, 'test/patient')
        self.test_dataset = NpyDataset(test_normal_dir, test_abnormal_dir, transform=transform)