import numpy as np

from metrics.metrics import accuracy, confusion_matrix
from model.decision_tree import ClassificationDT, RegressionDT
from datasets.digits_dataset import Digits
from config.decision_tree_config import cfg


def task_classification():
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
    pass


if __name__ == '__main__':
    np.random.seed(2000)

    task_classification()

