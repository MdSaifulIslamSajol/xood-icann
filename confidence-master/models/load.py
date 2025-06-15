import sys;

import pandas as pd

sys.path.insert(0, '..')  # Enable import from parent folder.
from models import cnn, resnet, densenet
import torch
from torchvision import transforms
from pathlib import Path
from utils import get_torch_device
from models.resnet_react import resnet50, resnet18, resnet34, resnet101
from models.transformer import Cifar10Transformer


def load_model(dataset, model):
    device = get_torch_device()
    if dataset == "imagenet":
        if model == "resnet18":
            m = resnet18(pretrained=True, num_classes=1000).to(device)
        elif model == "resnet34":
            m = resnet34(pretrained=True, num_c=1000).to(device)
        elif model == "resnet50":
            m = resnet50(pretrained=True, num_classes=1000).to(device)
        elif model == "resnet101":
            m = resnet101(pretrained=True, num_classes=1000).to(device)
        else:
            raise Exception(f"Unknown model: {model}")
        transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif model == "transformer":
        m = Cifar10Transformer(device)
        transform = transforms.Compose([])  # No input transform
    else:
        path = Path(__file__).parent / f"{dataset}/{model}/"
        state_dict = torch.load(path / "state_dict.pt", map_location=device)
        if model == "cnn":  # Small CNN, mostly for testing.
            m = cnn.ReluNet().to(device)
            m.load_state_dict(state_dict)
            transform = transforms.Compose([])
        elif model == "resnet":  # ResNet34
            m = resnet.ResNet34(100 if dataset == "cifar100" else 10).to(device)
            m.load_state_dict(state_dict, strict=False)
            transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        elif model == "densenet":  # DenseNet3
            m = densenet.DenseNet3(100, 100 if dataset == "cifar100" else 10).to(device)
            m.load_state_dict(state_dict, strict=False)
            transform = transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255),
                                             (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0))
        else:
            raise Exception(f"Unknown model: {model}")
        try:
            m.threshold = pd.read_csv(path / "percentiles.csv", index_col=0).loc[90].iloc[0]
        except FileNotFoundError:
            print("No threshold found.")
    return m.eval(), transform


def save_state_dict(path, name):
    device = torch.device("cpu")
    model = torch.load(path / name, device).eval()
    torch.save(model.state_dict(), path / "state_dict.pt")


if __name__ == "__main__":
    path = Path("cifar100/densenet")
    save_state_dict(path, "densenet_cifar100.pth")
