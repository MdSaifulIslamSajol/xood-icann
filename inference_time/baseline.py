import sys;

sys.path.insert(0, '..')
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from models.load import load_model
from utils import get_torch_device


def load_baseline(dataset, model):
    m, transform = load_model(dataset, model)
    device = get_torch_device()

    def predict(df):
        img_shape = (32, 32, 3)
        images = df["data"].to_numpy()
        images = images.reshape(images.shape[0], *img_shape)
        images = np.moveaxis(images, 3, 1)
        images = torch.tensor(images, dtype=torch.float)
        images = transform(images)
        images = TensorDataset(images)
        images = DataLoader(images, batch_size=128)
        output = []
        with torch.no_grad():
            for i, data in enumerate(images):
                data = data[0].to(device)
                out = m(data)
                output.append(out)
        output = torch.cat(output).cpu().detach().numpy()
        return np.max(output, axis=1)

    return predict
