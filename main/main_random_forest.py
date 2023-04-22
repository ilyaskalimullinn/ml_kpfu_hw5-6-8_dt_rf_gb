import os

import numpy as np

from config.decision_tree_config import cfg
from datasets.digits_dataset import Digits
from metrics.metrics import accuracy, confusion_matrix
from model.random_forest import ClassificationRandomForest
from utils.visualisation import Visualisation


if __name__ == "__main__":
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

    experiments = []

    L_1_arr = np.random.randint(MIN_L1, MAX_L1, FOREST_AMOUNT)
    L_2_arr = np.random.randint(MIN_L2, MAX_L2, FOREST_AMOUNT)
    M_arr = np.random.randint(MIN_M, MAX_M, FOREST_AMOUNT)
    for L_1, L_2, M in zip(L_1_arr, L_2_arr, M_arr):
        model = ClassificationRandomForest(M, MAX_DEPTH, MIN_ENTROPY, MIN_ELEMENTS_TERMINAL_NODE, L_1, L_2,
                                           CLASSES_AMOUNT)

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

    experiments.sort(key=lambda x: x["accuracy_valid"])

    for experiment in experiments[:BEST_MODELS_AMOUNT]:
        model = experiment["model"]
        experiment["accuracy_test"] = accuracy(
            model.get_prediction(dataset.inputs_test, return_probability_vector=False), dataset.targets_test
        )

    visualisation = Visualisation(graphs_dir=GRAPHS_DIR)

    visualisation.visualize_best_models(experiments[:BEST_MODELS_AMOUNT], file_name="best_models.html")

    best_model = experiments[0]["model"]
    conf_matrix = confusion_matrix(best_model.get_prediction(dataset.inputs_test, return_probability_vector=False),
                                   dataset.targets_test, dataset.k)

    visualisation.visualize_confusion_matrix(conf_matrix, file_name="confusion_matrix.html")
