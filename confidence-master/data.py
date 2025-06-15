import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage import filters
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import fashion_mnist, mnist, cifar10, cifar100 as cf100
from scipy.io import loadmat
import torchvision
import torch

img_shape = (32, 32, 3)
imagenet_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor()
])


def load_dataset(name):
    global img_shape
    if name == "cifar10":
        img_shape = (32, 32, 3)
        return load_data(cifar10.load_data())
    elif name == "svhn":
        img_shape = (32, 32, 3)
        return load_data(load_svhn_data())
    elif name == "mnist":
        img_shape = (28, 28, 1)
        return load_data(mnist.load_data())
    elif name == "cifar100":
        img_shape = (32, 32, 3)
        return cifar100()
    elif name == "imagenet":
        img_shape = (224, 224, 3)
        return imagenet_validation()
    else:
        raise Exception(f"Unknown dataset: {name}")


def scale_and_save_in_df(images, labels, scale=True):
    if scale:
        images = images / 255
    images = images.reshape((images.shape[0], -1))
    df = pd.DataFrame(images, columns=images.shape[1] * ["data"])
    df["label"] = labels
    return df


def load_data(datasets):
    (train_images, train_labels), (test_images, test_labels) = datasets
    test_images, val_images, test_labels, val_labels = train_test_split(test_images, test_labels,
                                                                        stratify=test_labels, test_size=0.5,
                                                                        random_state=42)
    train = scale_and_save_in_df(train_images, train_labels)
    val = scale_and_save_in_df(val_images, val_labels)
    test = scale_and_save_in_df(test_images, test_labels)
    train.name, val.name, test.name = "Train", "Val", "Test"
    return {"Train": train, "Val": val, "Test": test}


def get_images_and_labels(df: pd.DataFrame, labels=True, chw=False):
    images = df["data"].to_numpy()
    images = images.reshape(images.shape[0], *img_shape)
    if chw:
        images = np.moveaxis(images, 3, 1)
    if labels:
        return images, to_categorical(df["label"])
    else:
        return images


def quantize_pixels(data):
    return np.round(255 * data) / 255


def rotated(df: pd.DataFrame, plot=False):
    df_pos, df_neg = train_test_split(df, test_size=.5)
    pos, neg = df_pos["data"].to_numpy(), df_neg["data"].to_numpy()
    pos, neg = pos.reshape(pos.shape[0], *img_shape), neg.reshape(neg.shape[0], *img_shape)
    images = np.concatenate([np.rot90(pos, k=1, axes=(1, 2)), np.rot90(neg, k=-1, axes=(1, 2))], axis=0)
    if plot:
        fig, axs = plt.subplots(3, 3)
        for i, ax in enumerate(np.concatenate(axs)):
            ax.imshow(images[-i][:, :, 0], cmap="Greys")
            ax.set_title(df_neg["label"].iloc[-i])
        plt.savefig("rotated.png")
    images = images.reshape(images.shape[0], -1)
    df = pd.DataFrame(images, columns=images.shape[1] * ["data"])
    df["label"] = np.concatenate([df_pos["label"].to_numpy(), df_neg["label"].to_numpy()])
    return df


# Same as Taylor 2018
def uniform(n, dim):
    df = pd.DataFrame(np.random.uniform(0, 1, (n, dim)), columns=dim * ["data"])
    df["label"] = np.NaN
    return df


# Same as Taylor 2018
def gaussian(n, dim):
    df = pd.DataFrame(np.clip(np.random.normal(loc=.5, scale=1, size=(n, dim)), a_min=0, a_max=1),
                      columns=dim * ["data"])
    df["label"] = np.NaN
    return df


def gaussian_noise(images, var, a_min=0, a_max=1):
    return np.clip(images + np.random.normal(0, np.sqrt(var), images.shape), a_min=a_min, a_max=a_max)


def awgn(df: pd.DataFrame):
    print(f"Creating Noisy Set", flush=True)
    noisy = df.sample(frac=1).reset_index(drop=True)
    variance = np.linspace(0, 2 ** (1 / 4), 100) ** 4
    # variance = np.logspace(-3, 1, 20)
    noisy["data"] = np.concatenate([gaussian_noise(noisy["data"].loc[ids].to_numpy(), var) for ids, var in
                                    zip(np.array_split(noisy.index.to_numpy(), len(variance)), variance)], axis=0)
    return noisy


def mixed_up(df: pd.DataFrame):
    print(f"Creating Mixed Up Set", flush=True)
    df_mixed_up = df[["data", "label"]].sample(frac=1.).reset_index(drop=True)
    fracs = [.5, .6, .7, .8, .9]
    df_mixed_up["data"] = np.concatenate(
        [frac * df_mixed_up["data"].loc[ids].to_numpy() + (1 - frac) * df["data"].sample(len(ids)).to_numpy()
         for ids, frac in zip(np.array_split(df_mixed_up.index.to_numpy(), len(fracs)), fracs)],
        axis=0)
    return df_mixed_up


def uniform_mixed_up(df: pd.DataFrame):
    print(f"Creating Uniform Mixed Up Set", flush=True)
    df_mixed_up = df[["data", "label"]].sample(frac=1.).reset_index(drop=True)
    uniform = np.random.uniform(0, 1, (len(df), np.prod(img_shape)))
    fracs = np.linspace(0, 1, 50)
    df_mixed_up["data"] = np.concatenate(
        [frac * df_mixed_up["data"].loc[ids].to_numpy() + (1 - frac) * uniform[ids, :]
         for ids, frac in zip(np.array_split(df_mixed_up.index.to_numpy(), len(fracs)), fracs)],
        axis=0)
    return df_mixed_up


def blurry(df: pd.DataFrame):
    print(f"Creating Blurry Set", flush=True)
    df = df[["data", "label"]].sample(frac=1).reset_index(drop=True)
    images = df["data"].to_numpy().reshape(len(df), *img_shape)
    df["data"] = gaussian_blur(images).reshape(len(df), -1)
    return df


def gaussian_blur(images: np.ndarray, std_range=(.1, 5)):
    stds = np.linspace(*std_range, 20)
    ids = np.arange(len(images))
    img_list = []
    for ids, std in zip(np.array_split(ids, len(stds)), stds):
        img_list += [filters.gaussian(images[i], std, multichannel=True) for i in ids]
    return np.concatenate(img_list, axis=0)


def targeted(set_name):
    data_name = set_name + "_adver_targeted"
    data = pd.read_pickle(f"results/datasets/adversarial/{set_name}/{data_name}.pkl")
    data = data[["data"]]
    data["label"] = np.NaN
    return data


def non_targeted(set_name):
    data_name = set_name + "_adver_non_targeted"
    data = pd.read_pickle(f"results/datasets/adversarial/{set_name}/{data_name}.pkl")
    data = data[["data"]]
    data["label"] = np.NaN
    return data


def imagenet():
    # data = pd.read_pickle('out_of_distribution/ImageNet.pkl')
    data = pd.read_pickle("/data/saiful/confidence-master/datasets/ImageNet.pkl")

    data["label"] = np.NaN
    return data


def isun_scaled():
    # data = pd.read_pickle('out_of_distribution/iSUN.pkl')
    data = pd.read_pickle("/data/saiful/confidence-master/datasets/iSUN.pkl")

    data["label"] = np.NaN
    return data


def lsun_scaled():
    # data = pd.read_pickle('out_of_distribution/lSUN.pkl')
    data = pd.read_pickle("/data/saiful/confidence-master/datasets/lSUN.pkl")

    data["label"] = np.NaN
    return data


def lsun_resized():
    # data = pd.read_pickle('out_of_distribution/lSUN_resize.pkl')
    data = pd.read_pickle("/data/saiful/confidence-master/datasets/lSUN_resize.pkl")

    data["label"] = np.NaN
    return data


def imagenet_resized():
    # data = pd.read_pickle('out_of_distribution/imagenet_resize.pkl')
    data = pd.read_pickle("/data/saiful/confidence-master/datasets/imagenet_resize.pkl")

    data["label"] = np.NaN
    return data


def fashion():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    return scale_and_save_in_df(x_test, np.nan)


def cifar100():
    return load_data(cf100.load_data())


def cifar100_as_ood():
    df = cifar100()["Test"]
    df["label"] = np.nan
    return df


import os
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_imagenet_a_as_ood():
    print("load_imagenet_a_as_ood():")
    # Path to the ImageNet-A dataset root directory
    imagenet_a_dir = '/data/saiful/imagenet-a_n_o/imagenet-a/imagenet-a/'

    # Define the transform: Resize each image to 32x32 and convert to a float tensor in [0, 1]
    transform = transforms.Compose([
        transforms.Resize((32, 32)),     # Resize image to 32x32
        transforms.ToTensor(),           # Convert image to PyTorch tensor with shape [C, H, W] and pixel values in [0, 1]
    ])

    # Load the dataset using torchvision's ImageFolder (ignores class, which is fine for OOD)
    imagenet_a_dataset = datasets.ImageFolder(root=imagenet_a_dir, transform=transform)

    # Create a DataLoader for efficient batch processing
    data_loader = DataLoader(imagenet_a_dataset, batch_size=128, shuffle=False, num_workers=4)

    # List to collect all image data arrays
    all_image_arrays = []

    # Loop through each batch and process images
    for batch_images, _ in data_loader:
        # Rearrange tensor shape from [batch_size, channels, height, width] to [batch_size, height, width, channels]
        batch_images = batch_images.permute(0, 2, 3, 1).numpy()     # [batch_size, 32, 32, 3]
        # Flatten each image to a 1D vector of length 3072 (32*32*3)
        batch_images_flattened = batch_images.reshape(batch_images.shape[0], -1)   # [batch_size, 3072]
        all_image_arrays.append(batch_images_flattened)

    # Concatenate all image arrays to create a full dataset of shape [num_images, 3072]
    all_images_matrix = np.concatenate(all_image_arrays, axis=0)     # [num_images, 3072]

    # Create the DataFrame: each column is 'data', and add 'label' set to NaN
    column_names = [f'data' for _ in range(all_images_matrix.shape[1])]
    imagenet_a_dataframe = pd.DataFrame(all_images_matrix, columns=column_names)
    imagenet_a_dataframe['label'] = np.nan

    return imagenet_a_dataframe

def load_imagenet_o_as_ood():
    print("load_imagenet_o_as_ood():")
    # Path to the ImageNet-A dataset root directory
    imagenet_o_dir = '/data/saiful/imagenet-a_n_o/imagenet-o/imagenet-o/'

    # Define the transform: Resize each image to 32x32 and convert to a float tensor in [0, 1]
    transform = transforms.Compose([
        transforms.Resize((32, 32)),     # Resize image to 32x32
        transforms.ToTensor(),           # Convert image to PyTorch tensor with shape [C, H, W] and pixel values in [0, 1]
    ])

    # Load the dataset using torchvision's ImageFolder (ignores class, which is fine for OOD)
    imagenet_o_dataset = datasets.ImageFolder(root=imagenet_o_dir, transform=transform)

    # Create a DataLoader for efficient batch processing
    data_loader = DataLoader(imagenet_o_dataset, batch_size=128, shuffle=False, num_workers=4)

    # List to collect all image data arrays
    all_image_arrays = []

    # Loop through each batch and process images
    for batch_images, _ in data_loader:
        # Rearrange tensor shape from [batch_size, channels, height, width] to [batch_size, height, width, channels]
        batch_images = batch_images.permute(0, 2, 3, 1).numpy()     # [batch_size, 32, 32, 3]
        # Flatten each image to a 1D vector of length 3072 (32*32*3)
        batch_images_flattened = batch_images.reshape(batch_images.shape[0], -1)   # [batch_size, 3072]
        all_image_arrays.append(batch_images_flattened)

    # Concatenate all image arrays to create a full dataset of shape [num_images, 3072]
    all_images_matrix = np.concatenate(all_image_arrays, axis=0)     # [num_images, 3072]

    # Create the DataFrame: each column is 'data', and add 'label' set to NaN
    column_names = [f'data' for _ in range(all_images_matrix.shape[1])]
    imagenet_o_dataframe = pd.DataFrame(all_images_matrix, columns=column_names)
    imagenet_o_dataframe['label'] = np.nan

    return imagenet_o_dataframe

# # Usage Example:
# imagenet_a_ood_32_dataframe = load_imagenet_a_as_ood()
# print(imagenet_a_ood_32_dataframe.shape)   # e.g., (N, 3073)
# print(imagenet_a_ood_32_dataframe.head())

def load_svhn_data(test_only=False):
    # test = loadmat("models/svhn/test_32x32.mat")
    test = loadmat("/data/saiful/confidence-master/datasets/svhn/test_32x32.mat")
    x_test = test['X']
    x_test = np.moveaxis(x_test, -1, 0)
    y_test = test['y']
    y_test[y_test == 10] = 0
    if test_only:
        return x_test, y_test
    # train = loadmat("models/svhn/train_32x32.mat")
    train = loadmat("/data/saiful/confidence-master/datasets/svhn/train_32x32.mat")

    x_train = train['X']
    x_train = np.moveaxis(x_train, -1, 0)
    y_train = train['y']
    y_train[y_train == 10] = 0

    return (x_train, y_train), (x_test, y_test)


def imagenet_validation(debug=False):
    dataset = torchvision.datasets.ImageFolder(root="imagenet/dummy" if debug else "imagenet/imagenet_val",
                                               transform=imagenet_transform)
    loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=len(dataset))
    x = next(iter(loader))[0].numpy()
    x = np.moveaxis(x, 1, 3)
    y = np.array(dataset.targets)
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.5, random_state=42)
    train = scale_and_save_in_df(x_train, y_train, scale=False)
    test = scale_and_save_in_df(x_test, y_test, scale=False)
    return {"Train": train, "Val": train, "Test": test}


def svhn_as_ood():
    return scale_and_save_in_df(load_svhn_data(test_only=True)[0], np.nan)


def cifar10_as_ood():
    df = load_data(cifar10.load_data())["Test"]
    df["label"] = np.nan
    return df


def rotated_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    df = scale_and_save_in_df(x_test, y_test)
    df = rotated(df[df["label"] != 0])
    df["label"] = np.nan
    return df


def augmented(df):
    print(f"Creating Augmented Set", flush=True)
    df = df.sample(frac=1.0).reset_index(drop=True)
    generator = ImageDataGenerator(
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        brightness_range=[.2, 2],
        zoom_range=[.9, 1.1],
    )
    x = get_images_and_labels(df, labels=False)
    generator.fit(x)
    x, y = generator.flow(x, df["label"], batch_size=len(df)).next()
    df = df[["data", "label"]].copy()
    df["data"] = np.reshape(x / 255, (len(df), -1))
    df["label"] = y
    return df


def scale_and_shift(df, scale):
    x_min = np.linspace(*np.sort([0, 1 - scale]), 20)
    df["data"] = np.clip(np.concatenate([m + scale * df["data"].loc[ids].to_numpy() for ids, m in
                                         zip(np.array_split(df.index.to_numpy(), len(x_min)), x_min)], axis=0),
                         a_min=0., a_max=1.)
    return df


def shifted(df):
    print(f"Creating Shifted Set", flush=True)
    df = df.sample(frac=1).reset_index(drop=True)
    scale = np.logspace(-3, 3, 40, base=2)
    df["data"] = np.concatenate([scale_and_shift(df["data"].loc[ids], s) for ids, s in
                                 zip(np.array_split(df.index.to_numpy(), len(scale)), scale)], axis=0)
    return df


def calibration(df):
    cal_set = df[["data", "label"]].sample(frac=1.).reset_index(drop=True)

    def clean(x):
        x = x.reset_index(drop=True)
        return x

    def blurry_to_shifted(x):
        return shifted(blurry(x))

    def awgn_to_shifted(x):
        return shifted(awgn(x))
    # augmented = geometric, awgn_to_shifted =gaussian_noise, blurry = blurry_to_shifted
    mappings = clean, augmented, mixed_up, awgn_to_shifted, blurry_to_shifted
    # mappings = clean 
    # mappings = [clean]


    result = {}
    for f in mappings:
        df = f(cal_set)
        df["data"] = quantize_pixels(df["data"])
        result[f.__name__] = df
    return result


def distorted(df):
    df = df[["data", "label"]].sample(frac=1.).reset_index(drop=True)
    datasets = {
        "Clean": df,
        "Uniform MixUp": uniform_mixed_up(df),
        "Shifted": shifted(df),
        "Noisy": awgn(df),
        "Blurry": blurry(df),
        "Shifted -> Blurry": blurry(shifted(df)),
        "Shifted -> Noisy": awgn(shifted(df)),
        "Noisy -> Shifted": shifted(awgn(df)),
        "Augmented": augmented(df),
        "MixUp": mixed_up(df),
    }
    for dataset in datasets.values():
        dataset["data"] = quantize_pixels(dataset["data"])
    return datasets


def dtd():
    df = pd.read_pickle("out_of_distribution/dtd.pkl")
    df["label"] = np.NaN
    return df


def food():
    df = pd.read_pickle("out_of_distribution/food.pkl")
    df["label"] = np.NaN
    return df


def calt():
    df = pd.read_pickle("out_of_distribution/Calt.pkl")
    df["label"] = np.NaN
    return df


def CUB():
    df = pd.read_pickle("out_of_distribution/CUB.pkl")
    df["label"] = np.NaN
    return df


def places():
    df = pd.read_pickle("out_of_distribution/places.pkl")
    df["label"] = np.NaN
    return df


def stanford_dogs():
    df = pd.read_pickle("out_of_distribution/stanford_dogs.pkl")
    df["label"] = np.NaN
    return df


def out_of_dist(dataset_name, debug=False):
    datasets = {
        "Uniform": uniform(10000, np.prod(img_shape)),
        "Gaussian": gaussian(10000, np.prod(img_shape)),
    }
    if debug:
        return datasets
    if img_shape == (32, 32, 3):
        datasets.update({
            "TinyImageNet (Crop)": imagenet(),
            "TinyImageNet (Resize)": imagenet_resized(),
            "LSUN (Crop)": lsun_scaled(),
            "LSUN (Resize)": lsun_resized(),
            "iSUN": isun_scaled(),
            # "DTD": dtd(),
            # "Stanford Dogs": stanford_dogs(),
            # "food": food(),
            # "CUB": CUB(),
            # "Calt": calt(),
            # "places": places()
        })
    elif img_shape == (28, 28, 1):
        datasets.update({
            "Fashion MNIST": fashion(),
            "Rotated MNIST": rotated_mnist()
        })
    if dataset_name == "cifar10":
        datasets.update({
            "SVHN": svhn_as_ood(),
            "Cifar100": cifar100_as_ood(),
            "imagenet_a": load_imagenet_a_as_ood(),
            "imagenet_o": load_imagenet_o_as_ood()

        })
    elif dataset_name == "svhn":
        datasets.update({
            "Cifar10": cifar10_as_ood(),
            "Cifar100": cifar100_as_ood(),
            "imagenet_a": load_imagenet_a_as_ood(),
            "imagenet_o": load_imagenet_o_as_ood()
        })
    elif dataset_name == "cifar100":
        datasets.update({
            "SVHN": svhn_as_ood(),
            "Cifar10": cifar10_as_ood(),
            "imagenet_a": load_imagenet_a_as_ood(),
            "imagenet_o": load_imagenet_o_as_ood()
        })
    elif dataset_name == "imagenet":
        for dataset in ['Places', 'SUN', 'iNaturalist', 'DTD']:
            path = f"imagenet/DTD/images" if dataset == "DTD" else f"imagenet/{dataset}"
            ood = torchvision.datasets.ImageFolder(root=path,
                                                   transform=imagenet_transform)
            ood = torch.utils.data.DataLoader(ood, batch_size=len(ood), shuffle=True)
            x = next(iter(ood))[0].numpy()
            x = np.moveaxis(x, 1, 3)
            datasets[dataset] = scale_and_save_in_df(x, np.nan, scale=False)
            print("\n")
            print(dataset)
            print(datasets[dataset])
            print("\n")

    for name in datasets.keys():
        datasets[name]["data"] = quantize_pixels(datasets[name]["data"])
    return datasets


cifar100_classes = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

if __name__ == "__main__":
    print(imagenet_validation())
    exit()
    datasets = distorted(svhn_as_ood())
    datasets.update(out_of_dist("svhn"))
    for name, df in datasets.items():
        print(name)
        print("Max:", np.max(df["data"].to_numpy()))
        print("Min:", np.min(df["data"].to_numpy()))
