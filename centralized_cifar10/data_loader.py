from torchvision import datasets, transforms
import torch

def get_data_loaders(batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                std=(0.2470, 0.2435, 0.2616))
    ])
    train_set = datasets.CIFAR10(root='F:\BalajiAI\Data\CIFAR10', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root='F:\BalajiAI\Data\CIFAR10', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
