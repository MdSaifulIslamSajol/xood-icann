import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from confidenciator import *
from data import calibration, out_of_dist
import data
from models.load import load_model
import pandas as pd

debug = True

def label_list(dataset):
    if dataset == "svhn":
        return np.arange(10)
    if dataset == "cifar10":
        return data.cifar10_classes
    if dataset == "cifar100":
        return data.cifar100_classes
    raise KeyError(dataset)

class ConfusionTester:
    def __init__(self, dataset: str, ood_dataset: str):
        self.dataset = dataset
        self.data = data.load_dataset(dataset)
        self.conf: Confidenciator = None
        self.ood = data.load_dataset(ood_dataset)["Train"]
        if debug:
            self.ood = self.ood.sample(1000)
            self.data = {name: df.sample(1000) for name, df in self.data.items()}
        self.cal = None  # Training set for the logistic regression.

    def load_model(self, model, config):
        self.conf = Confidenciator(*load_model(self.dataset, model), self.data["Train"], **config)

    def create_table(self, scores: Dict):
        for score, f in scores.items():
            self.data["Test"][score] = f(self.data["Test"])
            self.ood[score] = f(self.ood)
            y_true = np.concatenate([np.ones(len(self.data["Test"])), np.zeros(len(self.ood))])
            y_pred = np.concatenate([self.data["Test"][score], self.ood[score]])
            print(f"{score} AUC:", round(100 * roc_auc_score(y_true, y_pred), 1))
        table = []
        for (label, pred), df in self.ood.groupby(["label", "pred"]):
            y_true = np.concatenate([np.ones(len(self.data["Test"])), np.zeros(len(df))])
            auc = [round(100 * roc_auc_score(y_true, np.concatenate([self.data["Test"][score], df[score]])), 1) for score in
                   scores.keys()]
            table.append([label, pred, len(df)] + auc)
        table = pd.DataFrame(table, columns=["Label", "Pred", "Count"] + [f"{score} AUC" for score in scores.keys()])
        return table

    def fit(self, c=None):
        if not self.cal:
            print("Creating Calibration Set", flush=True)
            self.cal = calibration(self.data["Val"])
        print("Fitting Logistic Regression", flush=True)
        self.conf.fit(self.cal, c=c)


def main():
    for in_dist, ood in [["cifar10", "cifar100"], ["cifar100", "cifar10"], ["cifar10", "svhn"]]:
        tester = ConfusionTester(in_dist, ood)
        tester.load_model("resnet", config={})
        tester.fit()
        scores = {
            "XOOD-M": tester.conf.predict_mahala,
            "XOOD-L": tester.conf.predict_proba,
            "SoftMax": tester.conf.softmax
        }
        table = tester.create_table(scores).sort_values("Count",ascending=False)
        ood_class_to_name = label_list(ood)
        id_class_to_name = label_list(in_dist)
        table["Label"] = [ood_class_to_name[i] for i in table["Label"]]
        table["Pred"] = [id_class_to_name[i] for i in table["Pred"]]
        print(table)
        table.to_csv(f"results/table_{in_dist=}_{ood=}.csv")


if __name__ == "__main__":
    debug = False
    main()
