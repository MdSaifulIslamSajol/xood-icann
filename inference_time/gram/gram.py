import random
import matplotlib.pyplot as plt
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable, grad
from torchvision import datasets, transforms
from torch.nn.parameter import Parameter
import gram.calculate_log as callog
import gram.load_resnet as load_resnet
import gram.load_densenet as load_densenet


def create_gram_detector(dataset, model):
    if model == "resnet":
        torch_model = load_resnet.load_resnet(dataset)
    elif model == "densenet":
        torch_model = load_densenet.load_densenet(dataset)

    batch_size = 128
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize

    ])
    transform_test = transforms.Compose([
        transforms.CenterCrop(size=(32, 32)),
        transforms.ToTensor(),
        normalize
    ])
    if dataset == "cifar10":
        data_train = list(torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=True, download=True,
                             transform=transform_test),
            batch_size=1, shuffle=False))
    elif dataset == "cifar100":
        data_train = list(torch.utils.data.DataLoader(
            datasets.CIFAR100('data', train=True, download=True,
                             transform=transform_test),
            batch_size=1, shuffle=False))

    train_preds = []
    train_confs = []
    train_logits = []
    for idx in range(0, len(data_train), 128):
        batch = torch.squeeze(torch.stack([x[0] for x in data_train[idx:idx + 128]]), dim=1).cuda()

        logits = torch_model(batch)
        confs = F.softmax(logits, dim=1).cpu().detach().numpy()
        preds = np.argmax(confs, axis=1)
        logits = (logits.cpu().detach().numpy())

        train_confs.extend(np.max(confs, axis=1))
        train_preds.extend(preds)
        train_logits.extend(logits)
    print("Done")

    test_preds = []
    test_confs = []
    test_logits = []
    if dataset == "cifar10":
        data = list(torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=False, download=True,
                             transform=transform_test),
            batch_size=1, shuffle=False))
    elif dataset == "cifar100":
        data = list(torch.utils.data.DataLoader(
            datasets.CIFAR100('data', train=False, download=True,
                              transform=transform_test),
            batch_size=1, shuffle=False))

    for idx in range(0, len(data), 128):
        batch = torch.squeeze(torch.stack([x[0] for x in data[idx:idx + 128]]), dim=1).cuda()

        logits = torch_model(batch)
        confs = F.softmax(logits, dim=1).cpu().detach().numpy()
        preds = np.argmax(confs, axis=1)
        logits = (logits.cpu().detach().numpy())

        test_confs.extend(np.max(confs, axis=1))
        test_preds.extend(preds)
        test_logits.extend(logits)
    print("Done")

    def detect(all_test_deviations, all_ood_deviations, verbose=True, normalize=True):
        average_results = {}
        for i in range(1, 11):
            random.seed(i)

            validation_indices = random.sample(range(len(all_test_deviations)), int(0.1 * len(all_test_deviations)))
            test_indices = sorted(list(set(range(len(all_test_deviations))) - set(validation_indices)))

            validation = all_test_deviations[validation_indices]
            test_deviations = all_test_deviations[test_indices]

            t95 = validation.mean(axis=0) + 10 ** -7
            if not normalize:
                t95 = np.ones_like(t95)
            test_deviations = (test_deviations / t95[np.newaxis, :]).sum(axis=1)
            ood_deviations = (all_ood_deviations / t95[np.newaxis, :]).sum(axis=1)

            results = callog.compute_metric(-test_deviations, -ood_deviations)
            for m in results:
                average_results[m] = average_results.get(m, 0) + results[m]

        for m in average_results:
            average_results[m] /= i
        if verbose:
            callog.print_results(average_results)
        return average_results

    def cpu(ob):
        for i in range(len(ob)):
            for j in range(len(ob[i])):
                ob[i][j] = ob[i][j].cpu()
        return ob

    def cuda(ob):
        for i in range(len(ob)):
            for j in range(len(ob[i])):
                ob[i][j] = ob[i][j].cuda()
        return ob

    class Detector:
        def __init__(self):
            self.all_test_deviations = None
            self.mins = {}
            self.maxs = {}
            self.t95 = None
            self.classes = range(100 if dataset == "cifar100" else 10)

        def compute_minmaxs(self, data_train, POWERS=[10]):
            for PRED in self.classes:
                train_indices = np.where(np.array(train_preds) == PRED)[0]
                train_PRED = torch.squeeze(torch.stack([data_train[i][0] for i in train_indices]), dim=1)
                mins, maxs = torch_model.get_min_max(train_PRED, power=POWERS)
                self.mins[PRED] = cpu(mins)
                self.maxs[PRED] = cpu(maxs)
                torch.cuda.empty_cache()

        def compute_test_deviations(self, POWERS=[10]):
            all_test_deviations = None
            test_classes = []
            for PRED in self.classes:
                test_indices = np.where(np.array(test_preds) == PRED)[0]
                test_PRED = torch.squeeze(torch.stack([data[i][0] for i in test_indices]), dim=1)
                test_confs_PRED = np.array([test_confs[i] for i in test_indices])

                test_classes.extend([PRED] * len(test_indices))

                mins = cuda(self.mins[PRED])
                maxs = cuda(self.maxs[PRED])
                test_deviations = torch_model.get_deviations(test_PRED, power=POWERS, mins=mins,
                                                             maxs=maxs) / test_confs_PRED[:, np.newaxis]
                cpu(mins)
                cpu(maxs)
                if all_test_deviations is None:
                    all_test_deviations = test_deviations
                else:
                    all_test_deviations = np.concatenate([all_test_deviations, test_deviations], axis=0)
                torch.cuda.empty_cache()
            self.all_test_deviations = all_test_deviations

            self.test_classes = np.array(test_classes)

        def compute_ood_deviations(self, ood, POWERS=[10]):
            ood_preds = []
            ood_confs = []

            for idx in range(0, len(ood), 128):
                batch = torch.squeeze(torch.stack([x[0] for x in ood[idx:idx + 128]]), dim=1).cuda()
                logits = torch_model(batch)
                confs = F.softmax(logits, dim=1).cpu().detach().numpy()
                preds = np.argmax(confs, axis=1)

                ood_confs.extend(np.max(confs, axis=1))
                ood_preds.extend(preds)
                torch.cuda.empty_cache()

            ood_classes = []
            all_ood_deviations = None
            for PRED in self.classes:
                ood_indices = np.where(np.array(ood_preds) == PRED)[0]
                if len(ood_indices) == 0:
                    continue
                ood_classes.extend([PRED] * len(ood_indices))

                ood_PRED = torch.squeeze(torch.stack([ood[i][0] for i in ood_indices]), dim=1)
                ood_confs_PRED = np.array([ood_confs[i] for i in ood_indices])
                mins = cuda(self.mins[PRED])
                maxs = cuda(self.maxs[PRED])
                ood_deviations = torch_model.get_deviations(ood_PRED, power=POWERS, mins=mins,
                                                            maxs=maxs) / ood_confs_PRED[
                                                                         :, np.newaxis]
                cpu(self.mins[PRED])
                cpu(self.maxs[PRED])
                if all_ood_deviations is None:
                    all_ood_deviations = ood_deviations
                else:
                    all_ood_deviations = np.concatenate([all_ood_deviations, ood_deviations], axis=0)
                torch.cuda.empty_cache()

            self.ood_classes = np.array(ood_classes)
            return all_ood_deviations
            # average_results = detect(self.all_test_deviations, all_ood_deviations)
            # return average_results, self.all_test_deviations, all_ood_deviations

        def compute_t95(self):
            validation_indices = random.sample(range(len(self.all_test_deviations)),
                                               int(0.1 * len(self.all_test_deviations)))
            validation = self.all_test_deviations[validation_indices]
            self.t95 = validation.mean(axis=0) + 10 ** -7

        def predict(self, ood):
            deviations = self.compute_ood_deviations(ood, POWERS=list(range(1, 11)))
            return (deviations / self.t95[np.newaxis, :]).sum(axis=1)

    detector = Detector()
    detector.compute_minmaxs(data_train, POWERS=list(range(1, 11)))
    detector.compute_test_deviations(POWERS=list(range(1, 11)))
    detector.compute_t95()

    def predict(df):
        img_shape = (32, 32, 3)
        images = df["data"].to_numpy()
        images = images.reshape(images.shape[0], *img_shape)
        images = np.moveaxis(images, 3, 1)
        images = torch.tensor(images, dtype=torch.float)
        images = normalize(images)
        images = torch.utils.data.TensorDataset(images)
        images = list(torch.utils.data.DataLoader(images, batch_size=1))
        return detector.predict(images)

    return predict
