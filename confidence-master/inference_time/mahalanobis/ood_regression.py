"""
Created on Sun Oct 21 2018
@author: Kimin Lee
"""
import numpy as np
import pandas as pd
from pathlib import Path

from mahalanobis import lib_regression
from mahalanobis.generate_mahalanobis  import prepare_mahalanobis

from sklearn.linear_model import LogisticRegressionCV


def create_mahala(df, dataset, model):
    outf = "mahalanobis/out/"
    Path(outf).mkdir(exist_ok=True)
    compute_mahalanobis = prepare_mahalanobis(dataset, model)
    # initial setup
    score_list = ['Mahalanobis_0.0', 'Mahalanobis_0.01', 'Mahalanobis_0.005', 'Mahalanobis_0.002', 'Mahalanobis_0.0014',
                  'Mahalanobis_0.001', 'Mahalanobis_0.0005']

    # train and measure the performance of Mahalanobis detector
    list_best_results, list_best_results_index = [], []
    list_best_results_out, list_best_results_index_out = [], []
    best_tnr, best_result, best_index = 0, 0, 0
    total_X, total_Y = compute_mahalanobis(df)
    for score in score_list:
        X_val, Y_val, X_test, Y_test = lib_regression.block_split(total_X, total_Y, outf)
        X_train = np.concatenate((X_val[:500], X_val[1000:1500]))
        Y_train = np.concatenate((Y_val[:500], Y_val[1000:1500]))
        X_val_for_test = np.concatenate((X_val[500:1000], X_val[1500:]))
        Y_val_for_test = np.concatenate((Y_val[500:1000], Y_val[1500:]))
        lr = LogisticRegressionCV(n_jobs=-1).fit(X_train, Y_train)
        y_pred = lr.predict_proba(X_train)[:, 1]
        # print('training mse: {:.4f}'.format(np.mean(y_pred - Y_train)))
        y_pred = lr.predict_proba(X_val_for_test)[:, 1]
        # print('test mse: {:.4f}'.format(np.mean(y_pred - Y_val_for_test)))
        results = lib_regression.detection_performance(lr, X_val_for_test, Y_val_for_test, outf)
        if best_tnr < results['TMP']['TNR']:
            best_tnr = results['TMP']['TNR']
            best_index = score
            best_result = lib_regression.detection_performance(lr, X_test, Y_test, outf)
            best_lr = lr
        list_best_results_out.append(best_result)
        list_best_results_index_out.append(best_index)
        list_best_results.append(list_best_results_out)
        list_best_results_index.append(list_best_results_index_out)

    def predict(df_: pd.DataFrame):
        x, y = compute_mahalanobis(df_)
        return best_lr.predict_proba(x)[:, 1]

    return predict
