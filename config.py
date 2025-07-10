import torch
import os
import torch.nn as nn


class Config:
    # 训练参数
    batch_size = 64
    epochs = 30
    learning_rate = 0.001
    val_ratio = 0.2  # 验证集比例
    patience = 10 #早停

    # 数据集路径
    processed_data_dir="./autodl-tmp"

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = 12

    #损失函数
    criterion = nn.CrossEntropyLoss()

    # 模型保存
    checkpoint_dir = "./autodl-tmp/checkpoints"
    best_model = "best_model.pth"

    # 日志
    log_dir = "./tf-logs"

    # 确保目录存在
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)