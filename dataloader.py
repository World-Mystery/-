from torch.utils.data import DataLoader
from configs.config import Config


def create_dataloaders(dataset, config):
    train_loader = DataLoader(
        dataset.train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers = config.num_workers
    )

    val_loader = DataLoader(
        dataset.val_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )

    test_loader = DataLoader(
        dataset.test_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )

    return train_loader, val_loader, test_loader