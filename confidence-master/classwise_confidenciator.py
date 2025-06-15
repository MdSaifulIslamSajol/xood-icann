import numpy as np
import pandas as pd
from torch.linalg import vector_norm
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data import get_images_and_labels
from typing import Callable, Dict
from utils import get_torch_device
from scipy.spatial.distance import mahalanobis
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PowerTransformer


class LpNorm:
    def __init__(self, p=2):
        self.p = p

    def __call__(self, x):
        return {f"L{self.p}-norm": vector_norm(x, self.p, dim=tuple(range(1, len(x.shape))))}


class DynamicRange:
    def __init__(self, p=1):
        self.p = p

    def __call__(self, x):
        x = x.view(x.shape[0], -1)
        return {"DynamicRange": torch.amax(torch.abs(x), dim=1) / (1e-15 + vector_norm(x, self.p, dim=1))}


class SplitDynamicRange:
    def __init__(self, p=1):
        self.p = p

    def __call__(self, x):
        def dr(y):
            y = y.view(x.shape[0], -1)
            return torch.amax(y, dim=1) / vector_norm(y, self.p, dim=1)

        return {"NegDynamicRange": dr(torch.relu(-x)),
                "PosDynamicRange": dr(torch.relu(x))}


class SplitLpNorm:
    def __init__(self, p=2):
        self.p = p

    def __call__(self, x):
        return {"NegLpNorm": vector_norm(torch.relu(-x), self.p, dim=tuple(range(1, len(x.shape)))),
                "PosLpNorm": vector_norm(torch.relu(x), self.p, dim=tuple(range(1, len(x.shape))))}


class Positivity:
    def __call__(self, x):
        return {"Positivity": torch.mean((x > 0).float(), dim=tuple(range(1, len(x.shape))))}


class Sum:
    def __call__(self, x):
        return {"Sum": torch.sum(x, dim=tuple(range(1, len(x.shape))))}


class MinMax:
    def __call__(self, x: torch.Tensor):
        amin, amax = torch.aminmax(x.view(x.shape[0], -1), dim=1)
        return {"Min": -amin, "Max": amax}


class FeatureExtractor(nn.Module):  # Based on https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904
    def __init__(self, model: nn.Module, transform):
        super().__init__()
        self.model = model
        self.feat_fns = [MinMax()]
        self._features = {}
        self.device = get_torch_device()
        self.transform = transform
        i = 0
        # print(model)
        supported_activations = nn.ReLU, nn.GELU, nn.LeakyReLU
        for layer in model.modules():
            if isinstance(layer, supported_activations):
                layer.register_forward_hook(self.save_features_hook(f"relu_{i}"))
                i += 1
                print(f"{layer} Is Relu {i}")

    def save_features_hook(self, layer_id: str) -> Callable:
        def fn(_, input_, __):
            for f in self.feat_fns:
                for name, output in f(input_[0]).items():
                    self._features[f"{name}_{layer_id}"] = output

        return fn

    def forward(self, x):
        output = self.model(x)
        return output, self._features

    def predict(self, images):
        images = torch.tensor(images, dtype=torch.float)
        images = self.transform(images)
        images = TensorDataset(images)
        images = DataLoader(images, batch_size=128)
        output = []
        features = {}
        with torch.no_grad():
            for i, data in enumerate(images):
                # print(f"Computing predictions: {i + 1}/{len(images)}             ", end="\r")
                # print("                                                          ", end="\r")
                data = data[0].to(self.device)
                out, feat = self(data)
                output.append(out)
                if len(features) == 0:
                    features = {key: [] for key in self._features.keys()}
                for k in features.keys():
                    features[k].append(feat[k])
            for k in features.keys():
                features[k] = torch.cat(features[k]).cpu().detach().numpy()
        return torch.cat(output).cpu().detach().numpy(), features


class Confidenciator:
    def __init__(self, model: nn.Module, transform, train_set: pd.DataFrame):
        self.model = FeatureExtractor(model, transform)
        self.feat_cols = []
        train_set = self.add_prediction_and_features(train_set)
        train_set = train_set[train_set["is_correct"]]
        self.reg = 10
        self.pt = {}
        # self.pt1 = PowerTransformer()
        # self.pt1.fit(train_set[self.feat_cols])
        self.inv_cov = {}
        self.mean = {}
        self.compute_mean_and_inv_cov(train_set)

    def compute_mean_and_inv_cov(self, dataset: pd.DataFrame):
        for label, df in dataset[self.feat_cols].groupby(dataset["label"]):
            pt = PowerTransformer()
            x = pt.fit_transform(df)
            # x = self.pt1.transform(df)
            cov = np.cov(x, rowvar=False)
            print(cov)
            cov = cov + self.reg * np.identity(len(self.feat_cols))
            self.pt[label] = pt
            self.inv_cov[label] = np.linalg.inv(cov)
            self.mean[label] = np.mean(x, axis=0)
        print(self.mean)

    def add_prediction_and_features(self, df: pd.DataFrame):
        pred, features = self.model.predict(get_images_and_labels(df, labels=False, chw=True))
        if len(self.feat_cols) == 0:
            self.feat_cols = ["Max_out", "Min_out"] + list(features.keys())
        df["pred"] = np.argmax(pred, axis=-1)
        df["is_correct"] = df["pred"] == df["label"].to_numpy()
        df["Max_out"] = np.max(pred, axis=-1)
        df["Min_out"] = -np.min(pred, axis=-1)
        df = pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)
        return df

    def predict_mahala(self, df: pd.DataFrame):
        dist = pd.Series(index=df.index)
        for label in np.unique(df["pred"]):
            idx = (df["pred"] == label)
            mean = self.mean[label]
            inv_cov = self.inv_cov[label]
            # x = self.pt1.transform(df[self.feat_cols].loc[idx])
            x = self.pt[label].transform(df[self.feat_cols].loc[idx])
            dist[idx] = -np.apply_along_axis(lambda row: mahalanobis(row, mean, inv_cov), 1, x)
        # print(f"Mahalanobis Distance {df.name}")
        # print(dist.groupby(df["pred"]).mean())
        return dist.to_numpy()


def split_features(features: np.ndarray):
    return np.concatenate([- np.clip(features, 0, None), - np.clip(-features, 0, None)], axis=1)
