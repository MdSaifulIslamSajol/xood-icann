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
from sklearn.preprocessing import PowerTransformer, StandardScaler
from torch.nn import functional as F


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
        return {f"DynamicRangeL{self.p}": torch.amax(torch.abs(x), dim=1) / (1e-15 + vector_norm(x, self.p, dim=1))}


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
        self.feat_fns = [LpNorm(2)]
        self._features = {}
        self.device = get_torch_device()
        self.transform = transform
        i = 0
        # print(model)
        for name, layer in model.named_modules():
            if "attention.output.dense" in name or "classifier.classifier" in name:
                layer.register_forward_hook(self.save_features_hook(f"{i}: {name}"))
                i += 1
                print(f"{i}: {name}")

    def save_features_hook(self, layer_id: str) -> Callable:
        def fn(_, input_, output_):
            for f in self.feat_fns:
                for name, output in f(output_).items():
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

    def predict_react(self, images, f=torch.logsumexp):
        images = torch.tensor(images, dtype=torch.float)
        images = self.transform(images)
        images = TensorDataset(images)
        images = DataLoader(images, batch_size=128)
        output = []
        with torch.no_grad():
            for i, data in enumerate(images):
                # print(f"Computing predictions: {i + 1}/{len(images)}             ", end="\r")
                # print("                                                          ", end="\r")
                data = data[0].to(self.device)
                logits = self.model.forward_threshold(data, threshold=1.0)
                # out = torch.nn.functional.softmax(logits, dim=1)
                out = f(logits.data, dim=1)
                output.append(out)
        return torch.cat(output).cpu().detach().numpy()

    def predict_f(self, images, f):
        images = torch.tensor(images, dtype=torch.float)
        images = self.transform(images)
        images = TensorDataset(images)
        images = DataLoader(images, batch_size=128)
        output = []
        with torch.no_grad():
            for i, data in enumerate(images):
                # print(f"Computing predictions: {i + 1}/{len(images)}             ", end="\r")
                # print("                                                          ", end="\r")
                data = data[0].to(self.device)
                logits = self.model.forward(data)
                out = f(logits, dim=1) if f else logits
                output.append(out)
        return torch.cat(output).cpu().detach().numpy()


class Confidenciator:
    def __init__(self, model: nn.Module, transform, train_set: pd.DataFrame):
        self.model = FeatureExtractor(model, transform)
        self.feat_cols = []
        train_set = self.add_prediction_and_features(train_set)
        train_set = train_set[train_set["is_correct"]]
        self.lr = None
        self.coeff = None
        self.pt = PowerTransformer()
        self.scaler = StandardScaler()
        self.cov = np.cov(self.pt.fit_transform(self.scaler.fit_transform(train_set[self.feat_cols])), rowvar=False)
        self.mean = np.zeros(len(self.feat_cols))
        self.reg = 10

    def add_prediction_and_features(self, df: pd.DataFrame):
        pred, features = self.model.predict(get_images_and_labels(df, labels=False, chw=True))
        if len(self.feat_cols) == 0:
            # self.feat_cols = ["Max_out", "Min_out"] + list(features.keys())
            self.feat_cols = list(features.keys())
        df["pred"] = np.argmax(pred, axis=-1)
        df["is_correct"] = df["pred"] == df["label"].to_numpy()
        # df["Max_out"] = np.max(pred, axis=-1)
        # df["Min_out"] = -np.min(pred, axis=-1)
        df = pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)
        return df

    def fit(self, cal: Dict[str, pd.DataFrame], c=None):
        nbr_folds = len(cal)
        cal = pd.concat(list(cal.values()), ignore_index=True)
        cal = self.add_prediction_and_features(cal)
        features = split_features(self.pt.transform(self.scaler.transform(cal[self.feat_cols])))
        self.lr = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(penalty="l2", solver="liblinear", class_weight="balanced"))
        ])
        if c is None:
            params = {
                "lr__C": list(np.logspace(-8, 0, 17)),
            }
            grid = GridSearchCV(self.lr, params, scoring='roc_auc', n_jobs=30, cv=nbr_folds)
            grid.fit(X=features, y=cal["is_correct"].to_numpy())
            self.lr = grid.best_estimator_
            print(pd.DataFrame(grid.cv_results_)[["mean_test_score", "std_test_score", "rank_test_score", "params"]])
        else:
            self.lr.fit(features, cal["is_correct"])
        self.coeff = pd.Series(self.lr["lr"].coef_[0],
                               [c + "+" for c in self.feat_cols] + [c + "-" for c in self.feat_cols])
        print(self.coeff)

    def predict_mahala(self, dataset: pd.DataFrame):
        if not all(col in dataset.columns for col in self.feat_cols):
            dataset = self.add_prediction_and_features(dataset)
        x = self.pt.transform(self.scaler.transform(dataset[self.feat_cols]))
        if self.reg < np.inf:
            inv_cov = np.linalg.inv(self.cov + self.reg * np.identity(len(self.feat_cols)))
            return -np.apply_along_axis(lambda row: mahalanobis(row, self.mean, inv_cov), 1, x)
        return -np.apply_along_axis(lambda row: np.linalg.norm(row - self.mean, ord=2), 1, x)

    def predict_proba(self, dataset: pd.DataFrame):
        if not all(col in dataset.columns for col in self.feat_cols):
            dataset = self.add_prediction_and_features(dataset)
        x = self.pt.transform(self.scaler.transform(dataset[self.feat_cols]))
        return self.lr.predict_proba(split_features(x))[:, 1]

    def react_energy(self, dataset: pd.DataFrame):
        pred = self.model.predict_react(get_images_and_labels(dataset, labels=False, chw=True))
        return pred

    def react_max(self, dataset: pd.DataFrame):
        pred = self.model.predict_react(get_images_and_labels(dataset, labels=False, chw=True), torch.amax)
        return pred

    def react_softmax(self, dataset: pd.DataFrame):
        pred = self.model.predict_react(get_images_and_labels(dataset, labels=False, chw=True), F.softmax)
        return np.max(pred, axis=-1)

    def energy(self, dataset: pd.DataFrame):
        pred = self.model.predict_f(get_images_and_labels(dataset, labels=False, chw=True), torch.logsumexp)
        return pred

    def softmax(self, dataset: pd.DataFrame):
        pred = self.model.predict_f(get_images_and_labels(dataset, labels=False, chw=True), F.softmax)
        return np.max(pred, axis=-1)

    def max(self, dataset: pd.DataFrame):
        pred = self.model.predict_f(get_images_and_labels(dataset, labels=False, chw=True), None)
        return np.max(pred, axis=-1)


def split_features(features: np.ndarray):
    return np.concatenate([- np.clip(features, 0, None), - np.clip(-features, 0, None)], axis=1)
