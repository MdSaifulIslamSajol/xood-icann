import time

import numpy as np
import pandas as pd
import sys
from gram.gram import create_gram_detector
import os
from xood import fit_x_ood
from baseline import load_baseline
from scipy import stats
from mahalanobis.ood_regression import create_mahala

sys.path.insert(0, '..')

import data as confidence_datasets


def inference_time(data, ood_detectors, dataset, model, n=10, conf=.99):
    result = pd.DataFrame(columns=["Inference Time (Seconds)", "Mean", "Std"], index=ood_detectors.keys())
    t_conf = stats.t.ppf(.5 + conf / 2, df=n - 1)
    for name, detector in ood_detectors.items():
        print(name, flush=True)
        t1 = time.time()
        detector(data)  # Skipping first round
        t2 = time.time()
        print("First round:", round(t2 - t1, 2))
        dt = []
        for _ in range(n):
            t1 = time.time()
            pred = detector(data)
            t2 = time.time()
            dt.append(t2 - t1)
        print(dt)
        mean = np.mean(dt)
        std = np.std(dt, ddof=1)
        result.loc[name, "Mean"] = mean
        result.loc[name, "Std"] = std
        result.loc[name, "Inference Time (Seconds)"] = f"{round(mean, 2)} ± {round(t_conf * std / np.sqrt(n), 4)}"
    baseline_mean = result["Mean"]["Baseline"]
    oh = 100 * (result["Mean"].to_numpy() - baseline_mean) / baseline_mean
    result.insert(1, "Overhead", [f"{round(x)}%" for x in oh])
    result = result.sort_values("Mean")
    print(result, flush=True)
    result.to_csv(f"inference_time_{dataset}_{model}_xood_last.csv")
    result = result[["Inference Time (Seconds)", "Overhead"]]
    with open(f'inference_time_{dataset}_{model}_xood_last.tex', 'w') as fo:
        result.to_latex(fo)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    data = confidence_datasets.gaussian(10000, 32 * 32 * 3)
    for dataset in "cifar10", "cifar100":
        for model in "resnet", "densenet":
            print("Creating Gram Detector")
            t1 = time.time()
            gram_detector = create_gram_detector(dataset, model)
            t2 = time.time()
            print(f"Created Gram detector in {round(t2 - t1, 4)} seconds", flush=True)

            print("Creating Mahalanobis Detector", flush=True)
            t1 = time.time()
            mahala = create_mahala(data, dataset, model)
            t2 = time.time()
            print(f"Created Mahalanobis detector in {round(t2 - t1, 4)} seconds", flush=True)
            print("Creating X-OOD Detector", flush=True)
            t1 = time.time()
            xood = fit_x_ood(dataset, model)
            t2 = time.time()
            print(f"Created X-OOD detector in {round(t2 - t1, 4)} seconds", flush=True)
            detectors = {
                "Baseline": load_baseline(dataset, model),
                "X-OOD L": xood.predict_proba,
                "X-OOD M": xood.predict_mahala,
                "Gram": gram_detector,
                "Mahalanobis": mahala,
            }
            inference_time(data, detectors, dataset, model)
