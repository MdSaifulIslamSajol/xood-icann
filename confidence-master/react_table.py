import pandas as pd
from pathlib import Path

pd.set_option('display.max_colwidth', None)


def map_columns(table):
    table["TNR (95% TPR)"] = 1 - table["FPR (95% TPR)"]
    table["Detection Acc."] = 1 - table["Detection Error"]
    return table


def boldJoin(nums):
    max_num = max(nums)
    nums_str = []
    for x in nums:
        num_str = str(x)
        if x < 10:
            num_str = " " + num_str
        elif x == 100:
            num_str = "100."
        if abs(max_num - x) <= 0.01:
            num_str = "\\textbf{%s}" % num_str
        nums_str.append(num_str)
    return "/".join(nums_str)


def collect_table(datasets, models, methods, oods):
    columns = ["TNR (95% TPR)", "Detection Acc.", "AUROC"]
    tables = []
    for dataset in datasets:
        for model in models:
            table = pd.DataFrame()
            summary = {}
            path = Path(f"results/{dataset}_{model}")
            for method in methods:
                summary[method] = map_columns(
                    pd.read_csv(path / f"summary_{method}.csv", header=0, index_col=0))
            idx = summary[methods[0]].index
            for col in columns:
                c = {}
                for ood in oods:
                    if ood in idx:
                        c[(dataset, model, ood)] = boldJoin([round(100 * summary[m][col].loc[ood], 1) for m in methods])
                table[col] = pd.Series(c)
            tables.append(table)
    return pd.concat(tables)


if __name__ == "__main__":
    methods = ["baseline", "energy", "react_energy", "x-ood-lr", "x-ood-mahala"]
    table = pd.concat([
        collect_table(
            datasets=["imagenet"],
            models=["resnet18", "resnet34", "resnet50", "resnet101"],
            methods=methods,
            oods=["Uniform", "Gaussian", "Places", "SUN", "iNaturalist", "DTD"]
        ),
        collect_table(
            datasets=["cifar10", "svhn", "cifar100"],
            models=["resnet", "densenet"],
            methods=methods,
            oods=["Uniform", "Gaussian", "TinyImageNet (Crop)", "TinyImageNet (Resize)", "LSUN (Crop)", "LSUN (Resize)", "iSUN", "SVHN",
                  "Cifar100", "cifar10"]
        ),
    ])
    table.to_latex("results/imagenet.tex", escape=False, multirow=False)
