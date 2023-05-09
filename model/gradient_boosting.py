import numpy as np

from model.decision_tree import RegressionDT


class RegressionDTGradientBoosting:
    def __init__(self, learning_rate: float = 0.1, weak_learners_amount: int = 10):
        self.weak_learners = []
        self.zero_learner = None
        self.learning_rate = learning_rate
        self.weak_learners_amount = weak_learners_amount

    def train(self, train_inputs: np.ndarray, train_targets: np.ndarray):
        self._init_zero_learner(train_targets)
        model_predictions = self.get_zero_learner_predictions(train_targets.shape)
        for i in range(self.weak_learners_amount):
            residuals = train_targets - model_predictions
            weak_learner = self._train_weak_learner(train_inputs, residuals)
            weak_learner_predictions = weak_learner.get_predictions(train_inputs).flatten()
            model_predictions = model_predictions + self.learning_rate * weak_learner_predictions
            self.weak_learners.append(weak_learner)

    def get_predictions(self, inputs: np.ndarray) -> np.ndarray:
        predictions = np.zeros(inputs.shape[0])
        for learner in self.weak_learners:
            predictions += learner.get_predictions(inputs).flatten()
        predictions *= self.learning_rate
        predictions += self.get_zero_learner_predictions(predictions.shape)
        return predictions

    def _init_zero_learner(self, targets: np.ndarray):
        self.zero_learner = targets.mean(axis=0)

    def _train_weak_learner(self, input_vectors: np.ndarray, residuals: np.ndarray) -> RegressionDT:
        model = RegressionDT(max_depth=1)
        model.train(input_vectors, residuals)
        return model

    def get_zero_learner_predictions(self, shape: tuple):
        return self.zero_learner * np.ones(shape=shape)
