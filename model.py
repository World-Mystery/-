import torch.nn as nn
import torch.nn.functional as F


class NetWork(nn.Module):
    def __init__(self, in_channel=1, expansion=2, num_classes=2, dropout_rate=0.3):
        super(NetWork, self).__init__()

        self.expansion = expansion

        # 卷积模块
        self.conv_layers = nn.Sequential(
            # Block 1
            self._make_conv_block(in_channel, 16 * expansion, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2),  # 64->32

            # Block 2
            self._make_conv_block(16 * expansion, 32 * expansion, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2),  # 32->16

            # Block 3
            self._make_conv_block(32 * expansion, 64 * expansion, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2),  # 16->8

            # Block 4 - 减少池化次数保持空间信息
            self._make_conv_block(64 * expansion, 128 * expansion, kernel_size=3, stride=1, padding=1),
        )

        # 自适应池化
        self.adaptive_pool = nn.AdaptiveAvgPool3d((4, 4, 4))

        # 全连接层：添加Dropout和批量归一化
        self.fc = nn.Sequential(
            nn.Linear(128 * expansion * 4 ** 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),

            nn.Linear(256, num_classes)
        )

    def _make_conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        """创建带残差连接的卷积块"""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # 卷积特征提取
        x = self.conv_layers(x)

        # 自适应池化适应不同尺寸
        x = self.adaptive_pool(x)

        # 展平特征
        x = x.view(x.size(0), -1)

        # 分类输出
        return self.fc(x)