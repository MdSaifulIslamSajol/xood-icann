from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, det_curve, average_precision_score, roc_curve
from tensorflow.keras.datasets import cifar10, mnist

from confidenciator_transformer import Confidenciator, split_features
from data import distorted, calibration, out_of_dist, load_data
import data
from data import load_svhn_data, imagenet_validation
from utils import binary_class_hist, df_to_pdf
from models.load import load_model


def taylor_scores(in_dist, out_dist):
    y_true = np.concatenate([np.ones(len(in_dist)), np.zeros(len(out_dist))])
    y_pred = np.concatenate([in_dist, out_dist])
    fpr, fnr, thr = det_curve(y_true, y_pred, pos_label=1)
    det_err = np.min((fnr + fpr) / 2)
    fpr, tpr, thr = roc_curve(y_true, y_pred)
    fpr95_sk = fpr[np.argmax(tpr >= .95)]
    scores = pd.Series({
        "FPR (95% TPR)": fpr95_sk,
        "Detection Error": det_err,
        "AUROC": roc_auc_score(y_true, y_pred),
        "AUPR In": average_precision_score(y_true, y_pred, pos_label=1),
        "AUPR Out": average_precision_score(y_true, 1 - y_pred, pos_label=0),
    })
    return scores


class FeatureTester:
    def __init__(self, dataset: str, model: str, name=""):
        self.dataset = dataset
        self.model = model
        data.img_shape = (32, 32, 3)
        if dataset == "cifar10":
            self.data = load_data(cifar10.load_data())
        elif dataset == "svhn":
            self.data = load_data(load_svhn_data())
        elif dataset == "mnist":
            data.img_shape = (28, 28, 1)
            self.data = load_data(mnist.load_data())
        elif dataset == "cifar100":
            self.data = data.cifar100()
        elif dataset == "imagenet":
            data.img_shape = (224, 224, 3)
            self.data = imagenet_validation()
        else:
            raise Exception(f"Unknown dataset: {dataset}")
        m, transform = load_model(dataset, model)
        # print(m)
        self.path = Path(f"results/{dataset}_{model}")
        self.path = (self.path / name) if name else self.path
        self.path.mkdir(exist_ok=True, parents=True)
        print("Creating Confidenciator", flush=True)
        self.conf = Confidenciator(m, transform, self.data["Train"])
        # self.conf.plot_model(self.path) TODO implement this.

        print("Adding Feature Columns")
        for name, df in self.data.items():
            self.data[name] = self.conf.add_prediction_and_features(self.data[name])
        self.compute_accuracy(self.data)
        print("Creating Out-Of-Distribution Sets", flush=True)
        self.ood = {name: self.conf.add_prediction_and_features(df) for name, df in out_of_dist(self.dataset).items()}
        self.cal = None  # Training set for the logistic regression.

    def compute_accuracy(self, datasets):
        try:
            accuracy = pd.read_csv(self.path / "accuracy.txt", sep=":", index_col=0)["Accuracy"]
        except FileNotFoundError:
            accuracy = pd.Series(name="Accuracy", dtype=float)
        for name, df in datasets.items():
            accuracy[name] = df["is_correct"].mean()
            print(f"Accuracy {name}: {accuracy[name]}")
        accuracy.sort_values(ascending=False).to_csv(self.path / "accuracy.txt", sep=":")
        print("Done", flush=True)

    def create_summary(self, f, name="", corr=False):
        print("Creating Taylor Table", flush=True)
        pred = {name: f(df) for name, df in self.ood.items()}
        pred_clean = f(self.data["Test"])
        all = np.concatenate(list(pred.values()) + [pred_clean])
        p_min, p_max = np.min(all), np.max(all)

        def map_pred(x):  # This function is used since some scores only support values between 0 and 1.
            return (x - p_min) / (p_max - p_min)

        pred["All"] = np.concatenate(list(pred.values()))
        table = pd.DataFrame.from_dict(
            {name: taylor_scores(map_pred(pred_clean), map_pred(p)) for name, p in pred.items()}, orient="index")
        table.to_csv(self.path / f"summary_{name}.csv")
        df_to_pdf(table, decimals=4, path=self.path / f"summary_{name}.pdf", vmin=0, percent=True)
        if corr:
            pred_corr = pred_clean[self.data["Test"]["is_correct"]]
            table = pd.DataFrame.from_dict(
                {name: taylor_scores(map_pred(pred_corr), map_pred(p)) for name, p in pred.items()}, orient="index")
            table.to_csv(self.path / f"summary_correct_{name}.csv")
            df_to_pdf(table, decimals=4, path=self.path / f"summary_correct_{name}.pdf", vmin=0, percent=True)

    def test_separation(self, test_set: pd.DataFrame, datasets: dict, name: str, split=False):
        if "All" not in datasets.keys():
            datasets["All"] = pd.concat(datasets.values()).reset_index(drop=True)
        summary_path = self.path / (f"{name}_split" if split else name)
        summary_path.mkdir(exist_ok=True, parents=True)
        summary = {dataset: {} for dataset in datasets.keys()}
        for feat in np.unique([c.split("_")[0] for c in self.conf.feat_cols]):
            feat_list = [f for f in self.conf.feat_cols if feat in f]
            if split & (feat != "Conf"):
                feat_list = list(sorted([f + "-" for f in feat_list] + [f + "+" for f in feat_list]))
            fig, axs = plt.subplots(len(datasets), len(feat_list), squeeze=False,
                                    figsize=(4 * len(feat_list) + 6, 5 * len(datasets)), sharex="col")
            for i, (dataset_name, dataset) in enumerate(datasets.items()):
                if dataset_name != "Clean":
                    dataset = pd.concat([dataset, test_set]).reset_index()
                feats = pd.DataFrame(self.conf.pt.transform(
                    self.conf.scaler.transform(dataset[self.conf.feat_cols])), columns=self.conf.feat_cols)
                if split:
                    cols = list(feats.columns)
                    feats = pd.DataFrame(split_features(feats.to_numpy()),
                                         columns=[c + "+" for c in cols] + [c + "-" for c in cols])
                for j, feat_id in enumerate(feat_list):
                    summary[dataset_name][feat_id] = binary_class_hist(feats[feat_id], dataset["is_correct"],
                                                                       axs[i, j],
                                                                       "", label_1="Clean", label_0=dataset_name)
            plt.savefig(summary_path / feat)
        if split:
            summary["LogReg Coeff"] = self.conf.coeff
        # save_corr_table(feature_table, self.path / f"corr_distorted", self.dataset_name)
        summary = pd.DataFrame(summary)
        summary.to_csv(f"{summary_path}.csv")
        df_to_pdf(summary, decimals=4, path=f"{summary_path}.pdf", vmin=0, percent=True)

    def fit(self, c=None):
        if not self.cal:
            print("Creating Calibration Set", flush=True)
            self.cal = calibration(self.data["Val"])
        print("Fitting Logistic Regression", flush=True)
        self.conf.fit(self.cal, c=c)

    def test_ood(self, split=False):
        print("\n==================   Testing features on Out-Of-Distribution Data   ==================\n",
              flush=True)
        self.test_separation(self.data["Test"].assign(is_correct=True), self.ood, "out_of_distribution", split)

    def test_distorted(self, split=False):
        print("\n=====================   Testing features on Distorted Data   =====================\n", flush=True)
        dist = distorted(self.data["Test"])
        dist = {name: self.conf.add_prediction_and_features(df) for name, df in dist.items()}
        self.compute_accuracy(dist)
        self.test_separation(self.data["Test"], dist, "distorted", split)


def test_features(dataset, model):
    print(f"\n\n================ Testing Features On {dataset} {model} ================", flush=True)
    ft = FeatureTester(dataset, model, "")
    ft.create_summary(ft.conf.predict_mahala, "x-ood-mahala")
    ft.test_ood()
    ft.fit()
    ft.create_summary(ft.conf.predict_proba, "x-ood-lr")


if __name__ == "__main__":
    test_features("cifar10", "transformer")

