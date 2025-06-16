import numpy as np
import pandas as pd
from models.load import load_model
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from typing import Callable
from utils import get_torch_device
import data
from pathlib import Path
from data import load_data, load_svhn_data
from tensorflow.keras.datasets import cifar10


class PenultimateExtractor(nn.Module):  # Based on https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904
    def __init__(self, model: nn.Module, transform):
        super().__init__()
        self.model = model
        self._features = None
        self.device = get_torch_device()
        self.transform = transform
        for layer in model.modules():
            if isinstance(layer, torch.nn.Linear):
                layer.register_forward_hook(self.save_features_hook())
                return

    def save_features_hook(self, ) -> Callable:
        def fn(_, input_, __):
            self._features = input_[0]

        return fn

    def forward(self, x):
        self.model(x)
        return self._features

    def extract_penultimate(self, images):
        images = torch.tensor(images, dtype=torch.float)
        images = self.transform(images)
        images = TensorDataset(images)
        images = DataLoader(images, batch_size=128)
        output = []
        with torch.no_grad():
            for i, im in enumerate(images):
                print(f"Computing activations: {i + 1}/{len(images)}             ", end="\r")
                print("                                                          ", end="\r")
                im = im[0].to(self.device)
                feat = self(im)
                output.append(feat)
        return torch.cat(output).cpu().detach().numpy()

    def compute_percentiles(self, images, path):
        activations = self.extract_penultimate(images)
        activations = np.reshape(activations, -1)
        percentiles = [80, 85, 90, 95, 99]
        percentiles = pd.Series(np.percentile(activations, percentiles), percentiles)
        percentiles.to_csv(path / "percentiles.csv")
        return percentiles


def compute_percentiles(dataset, models):
    print("Loading Dataset")
    images = data.load_dataset(dataset)["Train"]
    images = data.get_images_and_labels(images, labels=False, chw=True)
    for model in models:
        print(f"Computing Percentiles of {dataset} {model}")
        m, t = load_model(dataset, model)
        path = Path(f"models/{dataset}/{model}")
        path.mkdir(parents=True, exist_ok=True)
        pe = PenultimateExtractor(m, t)
        print(pe.compute_percentiles(images, path))


if __name__ == "__main__":
    for d in "svhn", "cifar10", "cifar100":
        compute_percentiles(d, ["resnet", "densenet"])
    exit()
    compute_percentiles("imagenet", ["resnet18", "resnet34", "resnet50", "resnet101"])

