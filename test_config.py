import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from confidenciator import *
from data import calibration, out_of_dist
import data
from models.load import load_model
import pandas as pd

ood_datasets = {
    "TinyImageNet (Crop)",
    "TinyImageNet (Resize)",
    "LSUN (Crop)",
    "LSUN (Resize)",
    "iSUN",
    "SVHN",
    'Places',
    'SUN',
    'iNaturalist',
    'DTD'
}

debug = True


class ConfigTester:
    def __init__(self, dataset: str):
        self.dataset = dataset
        self.data = data.load_dataset(dataset)
        self.conf = None
        self.ood = out_of_dist(self.dataset)
        self.ood = {name: self.ood[name] for name in ood_datasets.intersection(self.ood.keys())}
        print(self.ood)
        if debug:
            self.ood = {name: df.sample(100) for name, df in self.ood.items()}
            self.data = {name: df.sample(1000) for name, df in self.data.items()}
        self.cal = None  # Training set for the logistic regression.

    def load_model(self, model, config):
        self.conf = Confidenciator(*load_model(self.dataset, model), self.data["Train"], **config)

    def compute_score(self, f):
        pred = {name: f(df) for name, df in self.ood.items()}
        pred_clean = f(self.data["Test"])
        in_dist = pred_clean
        out_dist = np.concatenate(list(pred.values()))
        y_true = np.concatenate([np.ones(len(in_dist)), np.zeros(len(out_dist))])
        y_pred = np.concatenate([in_dist, out_dist])
        return 100 * roc_auc_score(y_true, y_pred)

    def fit(self, c=None):
        if not self.cal:
            print("Creating Calibration Set", flush=True)
            self.cal = calibration(self.data["Val"])
        print("Fitting Logistic Regression", flush=True)
        self.conf.fit(self.cal, c=c)


def test_configs(configs, test_lr=True, datasets=("cifar10", "svhn", "cifar100"), models=("resnet", "densenet")):
    result = pd.DataFrame()
    for dataset in datasets:
        print(f"\n\n======== {dataset} =========", flush=True)
        ct = ConfigTester(dataset)
        for model in models:
            print(f"\n\n======== {model} =========", flush=True)
            lr = {}
            mahala = {}
            for config_name, config in configs.items():
                print(config_name)
                ct.load_model(model, config)
                mahala[config_name] = ct.compute_score(ct.conf.predict_mahala)
                if test_lr:
                    ct.fit()
                    lr[config_name] = ct.compute_score(ct.conf.predict_proba)
            result[dataset, model, "mahala"] = pd.Series(mahala)
            if test_lr:
                result[dataset, model, "lr"] = pd.Series(lr)
    result.columns = pd.MultiIndex.from_tuples(result.columns)
    result = result.round(2)
    return result


def test_regularization():
    print(f"\n\n========Testing Regularization =========", flush=True)
    configs = {f"{c}": {"reg": c} for c in [0, .1, .3, 1, 3, 10, 30, 100]}
    configs["∞"] = {"reg": np.inf}
    plt.figure(figsize=(8, 3))
    dataset = "imagenet"
    models = ["resnet18", "resnet34", "resnet50", "resnet101"]
    result = test_configs(configs, False, [dataset], models)
    plt.subplot(1, 3, 3)
    plt.xlabel("C")
    plt.ylabel("AUROC")
    plt.title(dataset)
    for model in models:
        plt.plot(result.index.to_numpy(), result[dataset, model, "mahala"], ".-", label=model, alpha=.7)
    plt.legend(loc="lower right")
    plt.xticks(fontsize=8)

    result = test_configs(configs, test_lr=False)
    for i, model in enumerate(["resnet", "densenet"]):
        plt.subplot(1, 3, i + 1)
        plt.xlabel("C")
        plt.ylabel("AUROC")
        plt.title(model)
        for dataset in ["cifar10", "svhn", "cifar100"]:
            plt.plot(result.index.to_numpy(), result[dataset, model, "mahala"], ".-", label=dataset, alpha=.7)
        plt.legend(loc="lower right")
        plt.xticks(fontsize=8)

    plt.tight_layout(pad=.6)
    plt.savefig("results/regularization_test_img_net.pdf")
    result.to_csv("results/regularization_test_img_net.csv")


def bold_col_max(df: pd.DataFrame):
    for col in df.columns:
        i = df[col].argmax()
        df[col].iloc[i] = f"\\textbf{'{'}{df[col].iloc[i]}{'}'}"


def test_features():
    print(f"\n\n========Testing Features =========", flush=True)
    configs = {
        "Min & Max": {"features": (MinMax(),)},
        "Min": {"features": (Min(),)},
        "Max": {"features": (Max(),)},
        "Positivity": {"features": (Positivity(),)},
        "Sum": {"features": (Sum(),)},
        "L1": {"features": (LpNorm(1),)},
        "L2": {"features": (LpNorm(2),)},
        "L3": {"features": (LpNorm(3),)},
        "Split L1": {"features": (SplitLpNorm(1),)},
        "Split L2": {"features": (SplitLpNorm(2),)},
        "Split L3": {"features": (SplitLpNorm(3),)},
    }
    result = test_configs(configs, test_lr=not debug)
    result.to_csv("results/feature_selection.csv")
    bold_col_max(result)
    result.to_latex("results/feature_selection.tex", escape=False)


def test_additional_features():
    print(f"\n\n========Testing Features =========", flush=True)
    configs = {
        "Min & Max": {"features": (MinMax(),)},
        "Positivity": {"features": (MinMax(), Positivity(),)},
        "Sum": {"features": (MinMax(), Sum(),)},
        "L1": {"features": (MinMax(), LpNorm(1),)},
        "L2": {"features": (MinMax(), LpNorm(2),)},
        "L3": {"features": (MinMax(), LpNorm(3),)},
        "Split L1": {"features": (MinMax(), SplitLpNorm(1),)},
        "Split L2": {"features": (MinMax(), SplitLpNorm(2),)},
        "Split L3": {"features": (MinMax(), SplitLpNorm(3),)},
    }
    result = test_configs(configs, test_lr=not debug)
    result.to_csv("results/additional_features.csv")
    bold_col_max(result)
    result.to_latex("results/additional_features.tex", escape=False)


if __name__ == "__main__":
    debug = False
    test_regularization()
    test_features()
    test_additional_features()
