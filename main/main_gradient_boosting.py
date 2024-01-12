import os

import numpy as np

from config.decision_tree_config import cfg
from datasets.wine_dataset import WineDataset
from metrics.metrics import MSE
from model.gradient_boosting import RegressionDTGradientBoosting
from utils.visualisation import Visualisation

if __name__ == '__main__':
    np.random.seed(2000)

    wine = WineDataset(cfg)

    GRAPHS_DIR = os.path.normpath(os.path.join(os.path.abspath(os.path.curdir), "../graphs"))

    MIN_WEAK_LEARNERS_AMOUNT = 10
    MAX_WEAK_LEARNERS_AMOUNT = 30

    MIN_LEARNING_RATE = 0.05
    MAX_LEARNING_RATE = 2

    EXPERIMENTS_AMOUNT = 30
    BEST_MODELS_AMOUNT = 10

    weak_learners_amounts = np.random.randint(MIN_WEAK_LEARNERS_AMOUNT, MAX_WEAK_LEARNERS_AMOUNT,
                                              size=EXPERIMENTS_AMOUNT)
    learning_rates = np.random.uniform(MIN_LEARNING_RATE, MAX_LEARNING_RATE, size=EXPERIMENTS_AMOUNT)

    experiments = []
    for weak_learners_amount, learning_rate in zip(weak_learners_amounts, learning_rates):
        model = RegressionDTGradientBoosting(learning_rate=learning_rate, weak_learners_amount=weak_learners_amount)
        model.train(wine.inputs_train, wine.targets_train)
        predictions_valid = model.get_predictions(wine.inputs_valid)
        valid_mse = MSE(wine.targets_valid, predictions_valid)
        experiments.append({
            "model": model,
            "valid_mse": valid_mse,
            "weak_learners_amount": weak_learners_amount,
            "learning_rate": learning_rate
        })

    experiments.sort(key=lambda x: -x["valid_mse"])
    best_experiments = experiments[:BEST_MODELS_AMOUNT]

    for exp in best_experiments:
        exp["test_mse"] = MSE(wine.targets_test, exp["model"].get_predictions(wine.inputs_test))

    visualisation = Visualisation(GRAPHS_DIR)
    visualisation.visualize_best_gradient_boosting_models(best_experiments, file_name='best_gradient_boosting.html')
