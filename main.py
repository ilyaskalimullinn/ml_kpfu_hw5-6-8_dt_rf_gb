import numpy as np

from datasets.wine_dataset import WineDataset
from metrics.metrics import accuracy, confusion_matrix, MSE
from model.decision_tree import ClassificationDT, RegressionDT
from datasets.digits_dataset import Digits
from config.decision_tree_config import cfg


def task_classification():
    print("CLASSIFICATION")

    model = ClassificationDT(10, 10)
    digits = Digits(cfg)

    model.train(digits.inputs_train, digits.targets_train)

    predictions_test = model.get_predictions(digits.inputs_test, return_probability_vector=False)
    predictions_valid = model.get_predictions(digits.inputs_valid, return_probability_vector=False)

    accuracy_test = accuracy(predictions_test, digits.targets_test)
    accuracy_valid = accuracy(predictions_valid, digits.targets_valid)

    print("*"*50)
    print(f"Valid set accuracy is {round(accuracy_valid, 3)}, confusion matrix: ")
    print(confusion_matrix(predictions_valid, digits.targets_valid, 10))

    print("*" * 50)
    print(f"Test set accuracy is {round(accuracy_test, 3)}, confusion matrix: ")
    print(confusion_matrix(predictions_test, digits.targets_test, 10))


def task_regression():
    print("REGRESSION: ")

    wine = WineDataset(cfg)
    model = RegressionDT(10)
    model.train(wine.inputs_train, wine.targets_train)

    predictions_test = model.get_predictions(wine.inputs_test)
    predictions_valid = model.get_predictions(wine.inputs_valid)

    mse_test = MSE(predictions_test, wine.targets_test)
    mse_valid = MSE(predictions_valid, wine.targets_valid)

    print(f"Valid set mse: {round(mse_valid, 3)}; test set mse: {round(mse_test, 3)}")


if __name__ == '__main__':
    np.random.seed(2000)

    # task_classification()
    task_regression()

