import pandas as pd
import itertools
from pathlib import Path
import os
import numpy as np

pd.set_option('display.max_colwidth', None)

dataset_mapper = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "Cifar100": "CIFAR-100",
    "svhn": "SVHN",
    "SVHN": "SVHN",
    "Cifar10": "CIFAR-10",
    "TinyImageNet (Crop)": "TinyImgNet-C",
    "TinyImageNet (Resize)": "TinyImgNet-R",
    "iSUN": "iSUN",
    "LSUN (Crop)": "LSUN-C",
    "LSUN (Resize)": "LSUN-R"
}


def load_gram():
    cols1 = ['TNR', 'AUC', 'Err']
    cols2 = ['Baseline', 'ODIN', 'Mahalanobis', 'Ours']
    columns = [x for x in itertools.product(cols1, cols2)]
    columns.extend([("model", "name"), ("in_dist", "name"), ("out_dist", "name")])
    mux = pd.MultiIndex.from_tuples(columns)
    df = pd.read_csv("table.csv", header=[0, 1])
    df.cols = mux
    return df


def gram_ind(indist_model):
    indist, model = indist_model.split("_")
    indist = dataset_mapper[indist]
    return indist.upper(), model.upper()


def exists(file_path):
    f = Path(file_path)
    return f.is_file()


def generate(table, indist_model="cifar100_resnet", method="lr"):
    indist, model = gram_ind(indist_model)
    table = table[table[("model", "name")] == model]
    table = table[table[("in_dist", "name")] == indist]
    table = table.sort_values(('out_dist', 'name'))

    path = f"results/{indist_model}/summary_{method}"
    res = pd.read_csv(f"{path}.csv")
    method_mapper = {"lr": "LR", "mahala": "M"}
    method = method_mapper[method]
    first_column = res.iloc[:, 0]
    res.iloc[:, 0] = first_column.apply(lambda row: dataset_mapper.get(row, "None"))
    res = res[res.iloc[:, 0].isin(table[('out_dist', 'name')])]
    res = res.sort_values(res.columns[0])
    res["Detection Acc."] = (1 - res["Detection Error"]) * 100
    res["TNR (95% TPR)"] = (1 - res["FPR (95% TPR)"]) * 100
    res["AUROC"] = res["AUROC"] * 100
    res = res.round(1)
    ind_mapper = {"LR": [7, 12, 17], "M": [8, 14, 20]}
    inds = ind_mapper[method]
    table.insert(inds[0], ("TNR (95% TPR)", f"X-OOD {method}"), res["TNR (95% TPR)"].to_numpy())
    table.insert(inds[1], ("AUROC", f"X-OOD {method}"), res["AUROC"].to_numpy())
    table.insert(inds[2], ("Detection Acc.", f"X-OOD {method}"), res["Detection Acc."].to_numpy())
    return table


def generate_table(name):
    table = load_gram()
    for method in ["lr", "mahala"]:
        table = generate(table, name, method)
    if exists("fulltable.pkl"):
        fulltable = pd.read_pickle("fulltable.pkl")
        fulltable = pd.concat([fulltable, table])
    else:
        fulltable = table
    fulltable.to_pickle("fulltable.pkl")


def main():
    try:
        os.remove("fulltable.pkl")
    except Exception as e:
        print("Error: ", e)
    dirs = ["cifar10_resnet", "cifar10_densenet", "cifar100_densenet", "cifar100_resnet", "svhn_resnet",
            "svhn_densenet"]
    for d in dirs:
        generate_table(d)


def boldJoin(nums):
    max_num = max(nums)
    nums_str = []
    for x in nums:
        num_str = str(x)
        if abs(max_num - x) <= 0.2:
            num_str = "\\textbf{%s}" % num_str
        nums_str.append(num_str)
    return "/".join(nums_str)


def format_row(row):
    auc = row[["AUROC"]].to_numpy()
    tnr = row[["TNR (95% TPR)"]].to_numpy()
    detec_acc = row[["Detection Acc."]].to_numpy()
    auc = boldJoin(auc)
    tnr = boldJoin(tnr)
    detec_acc = boldJoin(detec_acc)
    return [auc, tnr, detec_acc]


def generate_latex(model_name):
    df = pd.read_pickle("fulltable.pkl")
    df = df[df[("model", "name")] == model_name]  # TODO: Do the same for resnet
    df = df.sort_values(by=[("model", "name"), ("in_dist", "name")])
    res = []
    for ind, row in df.iterrows():
        res.append(format_row(row))
    res = np.array(res)
    res = np.transpose(res)
    arrays = [df[("model", "name")], df[("in_dist", "name")], df[("out_dist", "name")]]
    index = pd.MultiIndex.from_arrays(arrays, names=('Model', 'In Dist', 'Out Dist'))
    data = {
        'TNR (95% TPR)': res[1],
        'AUROC': res[0],
        'Detection Acc.': res[2]
    }
    df = pd.DataFrame(data, index=index)
    df.to_csv(f"{model_name}.csv")
    with open(f"{model_name}.tex", "w") as fo:
        df.to_latex(fo, multirow=False, escape=False)


if __name__ == "__main__":
    main()
    for model_name in ["DENSENET", "RESNET"]:
        generate_latex(model_name)
