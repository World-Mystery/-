import torch
from torch.utils.tensorboard import SummaryWriter
from models.model import NetWork
from datasets.dataset import Dataset
from utils.dataloader import create_dataloaders
from utils.metrics import accuracy
from utils.utils import save_model, get_timestamp
from configs.config import Config
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import torch.optim as optim


def train_model():
    # 初始化配置
    config = Config()

    # 创建数据集和数据加载器
    dataset = Dataset(config)
    train_loader, val_loader, _ = create_dataloaders(dataset, config)

    # 初始化模型
    model = NetWork().to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = config.criterion

    # 动态学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        patience=3,
        factor=0.5,
        verbose=True
    )

    # 初始化梯度缩放器（混合精度训练）
    scaler = GradScaler()

    # 日志设置
    timestamp = get_timestamp()
    writer = SummaryWriter(f"{config.log_dir}/{timestamp}")

    # 训练参数
    best_val_acc = 0.0
    early_stop_counter = 0
    patience = config.patience

    # 在训练循环前添加
    '''
    print("测试数据加载器...")
    for i, (data, target) in enumerate(train_loader):
        print(f"成功加载批次 {i+1}, 数据: {data.shape}")
        if i > 1:  # 只测试2个批次
            break
    '''

    # 训练循环
    for epoch in range(1, config.epochs + 1):
        print(f'----------第{epoch}轮训练开始----------')
        model.train()
        train_loss = 0.0

        for data, target in train_loader:
            data, target = data.to(config.device), target.to(config.device)
            target = target.long()

            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                output = model(data)
                loss = criterion(output, target)

            # 使用梯度缩放器进行反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        # 计算训练损失
        train_loss /= len(train_loader)

        # 在验证集上评估
        val_loss, val_acc = evaluate(model, val_loader, criterion, config.device)

        # 学习率diaoduqi
        scheduler.step(val_acc)

        # 记录日志
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        print(f"Epoch {epoch}/{config.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = f"{config.checkpoint_dir}/{config.best_model}"
            save_model(model, optimizer, epoch, save_path)
            print(f"New best model saved with val acc: {val_acc:.4f}")

            early_stop_counter = 0  # 重置早停计数器
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"早停触发 epoch {epoch}")
                break

    writer.close()
    print("Training complete!")


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            target = target.long()

            output = model(data)
            total_loss += criterion(output, target).item()
            total_acc += accuracy(output, target)

    return total_loss / len(data_loader), total_acc / len(data_loader)


if __name__ == "__main__":
    train_model()