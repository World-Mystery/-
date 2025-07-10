import torch
import os
import time
from datetime import datetime

def save_model(model, optimizer, epoch, path):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, path)
    print(f"Model saved to {path}")

def load_model(model, optimizer, path, device):
    if os.path.isfile(path):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Model loaded from {path}")
        return checkpoint.get('epoch', 0)
    return 0

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")