from model import SimpleCNN
from train import train
from test import test
from data_loader import get_data_loaders
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    model = SimpleCNN().to(device)
    train_loader, test_loader = get_data_loaders(batch_size=32)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    num_epochs = 100
    for epoch in range(1, num_epochs + 1):
        train(model, train_loader, optimizer, criterion, device, epoch)
        if epoch % 10 == 0 or epoch == num_epochs:
            test(model, test_loader, device)

if __name__ == "__main__":
    main()
