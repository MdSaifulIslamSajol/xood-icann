import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_auc_score
import torch


def to_file_name(s: str):
    return s.casefold().replace(" ", "_")


def get_torch_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def df_to_pdf(df: pd.DataFrame, decimals: int, path, percent=True, vmin=-1, vmax=1):
    plt.clf()
    cmap = LinearSegmentedColormap.from_list('rg', ["r", "w", "g"], N=256)
    plt.figure(figsize=(((decimals + 2) * len(df.columns)) / 5, len(df.index) / 3))
    im = plt.matshow(df, alpha=1, aspect="auto", fignum=0, vmin=vmin, vmax=vmax, cmap=cmap)
    font = {'weight': 'bold',
            'size': 12}
    ax = plt.gca()
    for (i, j), x in np.ndenumerate(df.to_numpy()):
        x = str(round(100 * x, decimals - 2))[:decimals + 1] if percent else str(round(x, decimals))[:decimals + 2]
        ax.text(j, i, x, font, ha='center', va='center')
    plt.colorbar(im)
    plt.xticks(np.arange(len(df.columns)), df.columns, rotation=90)
    plt.yticks(np.arange(len(df.index)), df.index)
    plt.savefig(path, bbox_inches="tight")


def binary_class_hist(feature: pd.Series, label: pd.Series, ax, name, label_0="dirty", label_1="clean", bins=100,
                      x_range=None):
    auc = roc_auc_score(label, feature)
    if ax:
        if x_range is None:
            x_range = feature.min(), feature.max()
        ax.hist(feature[~ label], bins=bins, color="r", label=label_0, alpha=0.5, density=True, range=x_range)
        ax.hist(feature[label], bins=bins, color="g", label=label_1, alpha=0.5, density=True, range=x_range)
        ax.set_xlabel(name)
        # ax.set_ylabel("Density")
        # ax.title.set_text(f"{name} AUC: {round(100 * auc, 2)}%")
        ax.legend()
    return auc


def plot_model_performance(history, save_plt_path):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label="train")
    try:
        plt.plot(history['val_accuracy'], label="val")
    except KeyError:
        pass
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    # plt.ylim([0.8, 1.05])
    plt.grid(alpha=0.5)
    plt.legend(loc='upper left')

    plt.subplot(1, 2, 2)
    plt.semilogy(history['loss'], label="train")
    try:
        plt.semilogy(history['val_loss'], label="val")
    except KeyError:
        pass
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_plt_path)


def corr_feat_selection(df: pd.DataFrame, thresh=.95, method="pearson"):
    print(f"Selecting features with {method} correlation below {thresh}")
    corr = df.corr(method).abs()
    print(df)
    df_to_pdf(corr, 4, "corr.pdf")
    selected = []
    for i, feat in enumerate(corr.columns):
        if i == 0 or corr[feat].iloc[:i].max() < thresh:
            selected.append(feat)
    print(f"Selected {len(selected)}/{len(corr.columns)}:")
    print(selected)
    return selected
