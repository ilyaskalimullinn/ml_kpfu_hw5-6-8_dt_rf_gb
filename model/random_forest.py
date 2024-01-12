from typing import Optional

import numpy as np

from model.decision_tree import ClassificationDT


class ClassificationRandomForest:

    def __init__(self, nb_trees, max_depth, min_entropy, min_elem, nb_classes,
                 max_nb_dim_to_check=None, max_nb_thresholds=None, bagging_percent: Optional[float] = None):
        # bagging_percent: optional, if set, the bagging procedure will be performed. this parameter indicates
        # the amount of training set data that will be used to train one decision tree, e.g. if training set consists of
        # 1000 elements and bagging_percent is set to 0.3, then each tree will be trained on 300 random elements of
        # training set.
        self.nb_trees = nb_trees
        self.max_depth = max_depth
        self.min_entropy = min_entropy
        self.min_elem = min_elem
        self.max_nb_dim_to_check = max_nb_dim_to_check
        self.max_nb_thresholds = max_nb_thresholds
        self.nb_classes = nb_classes
        self.bagging_percent = bagging_percent

    def train(self, inputs, targets):
        self.trees = []
        if self.bagging_percent is not None:
            bagging_set_amount = int(targets.shape[0] * self.bagging_percent)
        for i in range(self.nb_trees):
            tree = ClassificationDT(self.max_depth, self.nb_classes, min_elem=self.min_elem, min_metric=self.min_entropy)
            if self.bagging_percent is None:
                tree.train(inputs, targets, (self.max_nb_dim_to_check, self.max_nb_thresholds))
            else:
                bagging_idx = np.random.randint(0, targets.shape[0], size=(bagging_set_amount))
                tree.train(inputs[bagging_idx], targets[bagging_idx])
            self.trees.append(tree)

    def get_prediction(self, inputs: np.ndarray, return_probability_vector: bool = True):
        """
        :param inputs: input vectors
        :param return_probability_vector: indicator whether to return vector of probability or just a class number
        :return: probability array or class number
        """
        predictions = np.zeros(shape=(inputs.shape[0], self.nb_classes))
        for tree in self.trees:
            predictions += tree.get_predictions(inputs)

        predictions = predictions / self.nb_trees

        if return_probability_vector:
            return predictions
        return predictions.argmax(axis=1)
