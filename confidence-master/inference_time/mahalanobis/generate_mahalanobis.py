"""
Created on Sun Oct 21 2018
@author: Kimin Lee
"""
import pandas as pd
import torch
from mahalanobis import data_loader
import numpy as np
from mahalanobis import lib_generation
from torch.utils.data import TensorDataset, DataLoader

from torch.autograd import Variable
import sys
from pathlib import Path

sys.path.insert(0, '..')

from models.load import load_model
from torchvision import transforms


def prepare_mahalanobis(dataset, net_type):
    # set the path to pre-trained model and output
    model, transform = load_model(dataset, net_type)
    in_transform = transforms.Compose([transforms.ToTensor(), transform])
    num_classes = 100 if dataset == "cifar100" else 10
    outf = "mahalanobis/out/"
    Path(outf).mkdir(exist_ok=True)
    # load dataset
    dataroot = 'mahalanobis/dataroot'
    Path(dataroot).mkdir(exist_ok=True, parents=True)
    train_loader, test_loader = data_loader.getTargetDataSet(dataset, 128, in_transform, dataroot)
    # set information about feature extaction
    model.eval()
    temp_x = torch.rand(2, 3, 32, 32).cuda()
    temp_x = Variable(temp_x)
    temp_list = model.feature_list(temp_x)[1]
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1

    print('get sample mean and covariance')
    sample_mean, precision = lib_generation.sample_estimator(model, num_classes, feature_list, train_loader)

    print('get Mahalanobis scores')
    m_list = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]
    for magnitude in m_list:
        print('Noise: ' + str(magnitude))
        for i in range(num_output):
            M_in = lib_generation.get_Mahalanobis_score(model, test_loader, num_classes, outf, \
                                                        True, net_type, sample_mean, precision, i, magnitude)
            M_in = np.asarray(M_in, dtype=np.float32)
            if i == 0:
                Mahalanobis_in = M_in.reshape((M_in.shape[0], -1))
            else:
                Mahalanobis_in = np.concatenate((Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1)
    Mahalanobis_in = np.asarray(Mahalanobis_in, dtype=np.float32)

    def compute_mahalanobis(df: pd.DataFrame):
        img_shape = (32, 32, 3)
        images = df["data"].to_numpy()
        images = images.reshape(images.shape[0], *img_shape)
        images = np.moveaxis(images, 3, 1)
        images = torch.tensor(images, dtype=torch.float)
        images = transform(images)
        images = TensorDataset(images)
        images = DataLoader(images, batch_size=128)
        for i in range(num_output):
            M_out = lib_generation.get_Mahalanobis_score(model, images, num_classes, outf, \
                                                         False, net_type, sample_mean, precision, i, magnitude)
            M_out = np.asarray(M_out, dtype=np.float32)
            if i == 0:
                Mahalanobis_out = M_out.reshape((M_out.shape[0], -1))
            else:
                Mahalanobis_out = np.concatenate((Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1)
        Mahalanobis_out = np.asarray(Mahalanobis_out, dtype=np.float32)
        Mahalanobis_data, Mahalanobis_labels = lib_generation.merge_and_generate_labels(Mahalanobis_out,
                                                                                        Mahalanobis_in)
        # mahalanobis_data = np.concatenate((Mahalanobis_data, Mahalanobis_labels), axis=1)
        return Mahalanobis_data, Mahalanobis_labels

    return compute_mahalanobis


if __name__ == '__main__':
    prepare_mahalanobis()
