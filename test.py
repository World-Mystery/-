import torch
import torch.nn as nn
from models.model import NetWork
from datasets.dataset import Dataset
from utils.dataloader import create_dataloaders
from utils.metrics import accuracy
from utils.utils import load_model
from configs.config import Config


def test_model():
    # 初始化配置
    config = Config()

    # 创建数据集和数据加载器
    dataset = Dataset(config)
    _, _, test_loader = create_dataloaders(dataset, config)

    # 初始化模型
    model = NetWork().to(config.device)
    optimizer = None  # 测试时不需要优化器

    # 加载最佳模型
    model_path = f"{config.checkpoint_dir}/{config.best_model}"
    load_model(model, optimizer, model_path, config.device)

    # 评估
    criterion = config.criterion
    test_loss, test_acc = evaluate(model, test_loader, criterion, config.device)

    print("\nTest Results:")
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")


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
    test_model()