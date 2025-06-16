import sys;
import cProfile

sys.path.insert(0, '..')
from confidenciator import Confidenciator
import data
from models.load import load_model
import os


def fit_x_ood(dataset, model) -> Confidenciator:
    m, transform = load_model(dataset, model)
    datasets = data.load_data(data.cifar10.load_data())
    conf = Confidenciator(m, transform, datasets["Train"])
    print("Creating Calibration Set", flush=True)
    cal = data.calibration(datasets["Val"])
    print("Fitting Logistic Regression", flush=True)
    conf.fit(cal)
    return conf


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    input_data = data.gaussian(10000, 32 * 32 * 3)

    print("Creating X-OOD Detector")
    xood = fit_x_ood()

    cProfile.run("xood.predict_proba(input_data)", sort='cumtime')
    print("\n\n\n\n")
    cProfile.run("xood.predict_mahala(input_data)", sort='cumtime')

