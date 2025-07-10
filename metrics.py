import torch

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        correct = torch.sum(pred == target).item()
        return correct / target.size(0)