import os

import numpy as np

from config.decision_tree_config import cfg
from datasets.digits_dataset import Digits
from metrics.metrics import accuracy, confusion_matrix
from model.random_forest import ClassificationRandomForest
from utils.visualisation import Visualisation


def visualize_experiments(experiments, best_experiments_amount, bagging=False):
    training_type = "bagging" if bagging else "rno"

    experiments.sort(key=lambda x: x["accuracy_valid"])

    for experiment in experiments[:best_experiments_amount]:
        model = experiment["model"]
        experiment["accuracy_test"] = accuracy(
            model.get_prediction(dataset.inputs_test, return_probability_vector=False), dataset.targets_test
        )

    visualisation = Visualisation(graphs_dir=GRAPHS_DIR)

    visualisation.visualize_best_rf_models(experiments[:best_experiments_amount],
                                           title=f"Best {best_experiments_amount} models, {training_type}",
                                           file_name=f"best_models_{training_type}.html", bagging=bagging)

    best_model = experiments[0]["model"]
    conf_matrix = confusion_matrix(best_model.get_prediction(dataset.inputs_test, return_probability_vector=False),
                                   dataset.targets_test, dataset.k)

    visualisation.visualize_confusion_matrix(conf_matrix, title=f"Confusion matrix, {training_type}",
                                             file_name=f"confusion_matrix_{training_type}.html")


def experiment_rno(dataset, models_amount, best_models_amount):
    experiments = []

    L_1_arr = np.random.randint(MIN_L1, MAX_L1, models_amount)
    L_2_arr = np.random.randint(MIN_L2, MAX_L2, models_amount)
    M_arr = np.random.randint(MIN_M, MAX_M, models_amount)
    for L_1, L_2, M in zip(L_1_arr, L_2_arr, M_arr):
        model = ClassificationRandomForest(M, MAX_DEPTH, MIN_ENTROPY, MIN_ELEMENTS_TERMINAL_NODE, CLASSES_AMOUNT, L_1,
                                           L_2)

        model.train(dataset.inputs_train, dataset.targets_train)

        accuracy_valid = accuracy(model.get_prediction(dataset.inputs_valid, return_probability_vector=False),
                                  dataset.targets_valid)

        experiments.append({
            "model": model,
            "accuracy_valid": accuracy_valid,
            "M": M,
            "L_1": L_1,
            "L_2": L_2
        })

    visualize_experiments(experiments, best_models_amount)


def experiment_bagging(dataset, models_amount, best_models_amount):
    experiments = []

    M_arr = np.random.randint(MIN_M, MAX_M, models_amount)
    bagging_percent_arr = np.random.uniform(size=(models_amount))

    for M, bagging_percent in zip(M_arr, bagging_percent_arr):
        model = ClassificationRandomForest(M, MAX_DEPTH, MIN_ENTROPY, MIN_ELEMENTS_TERMINAL_NODE,
                                           CLASSES_AMOUNT, bagging_percent=bagging_percent)

        model.train(dataset.inputs_train, dataset.targets_train)

        accuracy_valid = accuracy(model.get_prediction(dataset.inputs_valid, return_probability_vector=False),
                                  dataset.targets_valid)

        experiments.append({
            "model": model,
            "accuracy_valid": accuracy_valid,
            "M": M,
            "bagging_percent": bagging_percent
        })

    visualize_experiments(experiments, best_models_amount, bagging=True)


if __name__ == "__main__":
    np.random.seed(2000)

    FOREST_AMOUNT = 30
    BEST_MODELS_AMOUNT = 10
    # L1 - max number of dimensions
    MIN_L1 = 1
    MAX_L1 = 10
    # L2 - max number of thresholds
    MIN_L2 = 10
    MAX_L2 = 40
    # M - number of trees
    MIN_M = 5
    MAX_M = 15

    # other params
    MAX_DEPTH = 10
    MIN_ELEMENTS_TERMINAL_NODE = 5
    MIN_ENTROPY = 0
    CLASSES_AMOUNT = 10

    # params not related to the model
    GRAPHS_DIR = os.path.normpath(os.path.join(os.path.abspath(os.path.curdir), "../graphs"))

    dataset = Digits(cfg)

    training_type = input("Please, select training type: RNO or bagging: ").strip().lower()

    if training_type == "rno":
        print(f"Training {FOREST_AMOUNT} models using RNO...")
        experiment_rno(dataset, FOREST_AMOUNT, BEST_MODELS_AMOUNT)
    elif training_type == "bagging":
        print(f"Training {FOREST_AMOUNT} models using bagging...")
        experiment_bagging(dataset, FOREST_AMOUNT, BEST_MODELS_AMOUNT)
    else:
        print("Invalid training type")
