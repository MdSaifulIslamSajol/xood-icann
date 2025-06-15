from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, det_curve, average_precision_score, roc_curve
from tensorflow.keras.datasets import cifar10, mnist

from confidenciator import Confidenciator, split_features
from data import distorted, calibration, out_of_dist, load_data, load_svhn_data, imagenet_validation
import data
from utils import binary_class_hist, df_to_pdf
from models.load import load_model
from diptest import diptest

import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"TensorFlow has detected {len(gpus)} GPU(s):")
    for gpu in gpus:
        print(gpu)
else:
    print("No GPUs detected by TensorFlow.")


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
        self.data = data.load_dataset(dataset)
        m, transform = load_model(dataset, model)
        # print(m)
        self.path = Path(f"results/{dataset}_{model}_dip2") #folder
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
                                    figsize=(2 * len(feat_list) + 3, 2.5 * len(datasets)), sharex="col")
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
                                                                       axs[i, j], "", bins=50,
                                                                       label_1="ID", label_0=dataset_name)
            for ax, col in zip(axs[0], feat_list):
                ax.set_title(f"Layer {col}")

            for ax, row in zip(axs[:, 0], datasets.keys()):
                ax.set_ylabel(row, size='large')
            plt.tight_layout(pad=.4)
            plt.savefig(summary_path / f"{feat}.pdf")
        if split:
            summary["LogReg Coeff"] = self.conf.coeff
        # save_corr_table(feature_table, self.path / f"corr_distorted", self.dataset_name)
        summary = pd.DataFrame(summary)
        summary.to_csv(f"{summary_path}.csv")
        df_to_pdf(summary, decimals=4, path=f"{summary_path}.pdf", vmin=0, percent=True)
        
    def test_separation_for_dip_test(self, test_set: pd.DataFrame, datasets: dict, name: str, split=False):
        if "All" not in datasets.keys():
            datasets["All"] = pd.concat(datasets.values()).reset_index(drop=True)
        summary_path = self.path / (f"dip_{name}_split" if split else name)
        summary_path.mkdir(exist_ok=True, parents=True)
        summary = {dataset: {} for dataset in datasets.keys()}
        for feat in np.unique([c.split("_")[0] for c in self.conf.feat_cols]):
            feat_list = [f for f in self.conf.feat_cols if feat in f]
            if split & (feat != "Conf"):
                feat_list = list(sorted([f + "-" for f in feat_list] + [f + "+" for f in feat_list]))
            fig, axs = plt.subplots(len(datasets), len(feat_list), squeeze=False,
                                    figsize=(2 * len(feat_list) + 3, 2.5 * len(datasets)), sharex="col")
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
                    # summary[dataset_name][feat_id] = binary_class_hist(feats[feat_id], dataset["is_correct"],
                    #                                                    axs[i, j], "", bins=50,
                    #                                                    label_1="ID", label_0=dataset_name)
                    
                    # data will be add_features_data
                    dip, p_value = diptest(data)

                    print(f"Dip Statistic: {dip:.4f}")
                    print(f"P-Value: {p_value:.4f}")

                    # Plot histogram
                    plt.hist(data, bins=30, edgecolor='black', alpha=0.7)
                    plt.title("Histogram of Sample Data")
                    plt.xlabel("Value")
                    plt.ylabel("Frequency")
                    plt.show()
            for ax, col in zip(axs[0], feat_list):
                ax.set_title(f"Layer {col}")

            for ax, row in zip(axs[:, 0], datasets.keys()):
                ax.set_ylabel(row, size='large')
            plt.tight_layout(pad=.4)
            plt.savefig(summary_path / f"dip_{feat}.pdf")
        if split:
            summary["LogReg Coeff"] = self.conf.coeff
        # save_corr_table(feature_table, self.path / f"corr_distorted", self.dataset_name)
        summary = pd.DataFrame(summary)
        summary.to_csv(f"dip_{summary_path}.csv")
        df_to_pdf(summary, decimals=4, path=f"dip_{summary_path}.pdf", vmin=0, percent=True)
        
    def dip_test_only2(self, id_test_set: pd.DataFrame, datasets: dict, name: str, split=False):
        if "All" not in datasets.keys():
            datasets["All"] = pd.concat(datasets.values()).reset_index(drop=True)
    
        summary_path = self.path / (f"{name}_split" if split else name)
        summary_path.mkdir(exist_ok=True, parents=True)
    
        dip_summary = []
    
        for dataset_name, dataset in datasets.items():
            
            if dataset_name != "Clean":
                print("flag 1.112 ..")
                dataset = pd.concat([dataset, id_test_set]).reset_index()
    
            feats = pd.DataFrame(self.conf.pt.transform(
                self.conf.scaler.transform(dataset[self.conf.feat_cols])),
                columns=self.conf.feat_cols)
    
            if split:
                cols = list(feats.columns)
                feats = pd.DataFrame(split_features(feats.to_numpy()),
                                     columns=[c + "+" for c in cols] + [c + "-" for c in cols])
            flat_feats = feats.to_numpy().flatten()
            dip, pval = diptest(flat_feats)
    
            dip_summary.append({
                "Dataset": dataset_name,
                "Dip": dip,
                "p-value": pval
            })
    
            plt.figure(figsize=(6, 4))
            plt.hist(flat_feats, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
            plt.title(f"{dataset_name} Dip Test\nDip={dip:.4f}, p={pval:.4f}")
            plt.xlabel("Feature Value (Flattened)")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(summary_path / f"{dataset_name}_dip_hist.pdf")
            plt.close()
    
        dip_df = pd.DataFrame(dip_summary)
        dip_df.to_csv(summary_path / "diptest_results.csv", index=False)
            
    def dip_test_only(self, id_test_set: pd.DataFrame, datasets: dict, name: str, split=False):
        """
        Pairwise histogram comparison: ID vs each OOD dataset.
        """
        if "All" not in datasets.keys():
            datasets["All"] = pd.concat(datasets.values()).reset_index(drop=True)
    
        summary_path = self.path / (f"{name}_pairwise_split" if split else f"{name}_pairwise")
        summary_path.mkdir(exist_ok=True, parents=True)
    
        dip_summary = []
    
        # Extract ID features once (no need to repeat for each OOD)
        feats_id = pd.DataFrame(self.conf.pt.transform(
            self.conf.scaler.transform(id_test_set[self.conf.feat_cols])),
            columns=self.conf.feat_cols)
    
        if split:
            cols = list(feats_id.columns)
            feats_id = pd.DataFrame(split_features(feats_id.to_numpy()),
                                 columns=[c + "+" for c in cols] + [c + "-" for c in cols])
    
        flat_feats_id = feats_id.to_numpy().flatten()
    
        # Compute dip for ID itself (optional)
        dip_id, pval_id = diptest(flat_feats_id)
        dip_summary.append({
            "Dataset": "Clean",
            "Dip": dip_id,
            "p-value": pval_id
        })
    
        #### Now for each OOD dataset
        for dataset_name, dataset in datasets.items():
            if dataset_name == "Clean":
                continue  # Skip comparing ID to ID
    
            dataset = pd.concat([dataset, id_test_set]).reset_index()
    
            feats_ood = pd.DataFrame(self.conf.pt.transform(
                self.conf.scaler.transform(dataset[self.conf.feat_cols])),
                columns=self.conf.feat_cols)
    
            if split:
                cols = list(feats_ood.columns)
                feats_ood = pd.DataFrame(split_features(feats_ood.to_numpy()),
                                     columns=[c + "+" for c in cols] + [c + "-" for c in cols])
    
            flat_feats_ood = feats_ood.to_numpy().flatten()
    
            # Dip test for OOD
            dip_ood, pval_ood = diptest(flat_feats_ood)
            dip_summary.append({
                "Dataset": dataset_name,
                "Dip": dip_ood,
                "p-value": pval_ood
            })
    
            #### Pairwise histogram: ID vs OOD
            plt.figure(figsize=(7, 5))
            plt.hist(flat_feats_id, bins=100, color='green', alpha=0.5, label='ID', density=True)
            plt.hist(flat_feats_ood, bins=100, color='red', alpha=0.5, label=f'OOD: {dataset_name}', density=True)
            plt.title(f"ID vs {dataset_name}\nID Dip={dip_id:.4f}, OOD Dip={dip_ood:.4f}")
            plt.xlabel("Feature Value (Flattened)")
            plt.ylabel("Density")
            plt.legend()
            plt.tight_layout()
            plt.savefig(summary_path / f"ID_vs_{dataset_name}_dip_hist.pdf")
            plt.close()
    
        # Save full dip test summary
        dip_df = pd.DataFrame(dip_summary)
        dip_df.to_csv(summary_path / "pairwise_diptest_results.csv", index=False)
    


    def fit(self, c=None):
        if not self.cal:
            print("Creating Calibration Set", flush=True)
            self.cal = calibration(self.data["Val"])


        print("Fitting Logistic Regression", flush=True)
        self.conf.fit(self.cal, c=c)

    def test_ood2(self, split=False):
        print("\n==================   Testing features on Out-Of-Distribution Data   ==================\n",
              flush=True)
        self.test_separation(self.data["Test"].assign(is_correct=True), self.ood, "out_of_distribution", split)

    def test_distorted(self, split=False):
        print("\n=====================   Testing features on Distorted Data   =====================\n", flush=True)
        dist = distorted(self.data["Test"])
        dist = {name: self.conf.add_prediction_and_features(df) for name, df in dist.items()}
        self.compute_accuracy(dist)
        self.test_separation(self.data["Test"], dist, "distorted", split)

    def plot_detection(self, f, name):
        path = self.path / f"detection/{name}"
        path.mkdir(exist_ok=True, parents=True)
        pred = {name: f(df) for name, df in self.ood.items()}
        pred_clean = f(self.data["Test"])
        plt.figure(figsize=(4, 3))
        for key, p in pred.items():
            plt.clf()
            labels = pd.Series(np.concatenate([np.ones(len(pred_clean), dtype=bool), np.zeros(len(p), dtype=bool)]))
            p = pd.Series(np.concatenate([pred_clean, p]))
            binary_class_hist(p, labels, plt.gca(), name, label_0="OOD", label_1="ID")
            plt.tight_layout()
            plt.savefig(path / f"{key}.pdf")


def test_ood(dataset, model):
    print(f"\n\n================ Testing Features On {dataset} {model} ================", flush=True)
    ft = FeatureTester(dataset, model, "")
    # ft.create_summary(ft.conf.predict_mahala, "x-ood-mahala")
    # ft.create_summary(ft.conf.softmax, "baseline")
    # ft.create_summary(ft.conf.energy, "energy")
    # ft.create_summary(ft.conf.react_energy, "react_energy")
    ft.fit()
    # Add dip test directly here
    # ft.dip_test_only(
    #     id_test_set=ft.data["Test"].assign(is_correct=True),  # ID test set
    #     datasets=ft.ood,  # OOD datasets
    #     name="ood_dip_test",  # just a name to save results
    #     split=False  # you can make split=True if you want to use split features
    # )
  
    ft.create_summary(ft.conf.predict_proba, "x-ood-lr")
    # ft.test_ood2()


if __name__ == "__main__":
    test_ood("cifar10","resnet")
    # test_ood("cifar100","resnet")
    # test_ood("svhn","resnet")
    # test_ood("cifar10","densenet")
    # test_ood("cifar100","densenet")
    # test_ood("svhn","densenet")



    # for m in "resnet", "densenet":
    #     for d in "svhn", "cifar10", "cifar100":
    #         test_ood(d, m)
    # for m in "resnet18", "resnet34", "resnet50", "resnet101":
    #     test_ood("imagenet", m)
    print("Execution Finished..")
    
