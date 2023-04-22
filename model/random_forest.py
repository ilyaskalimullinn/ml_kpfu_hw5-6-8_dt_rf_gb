import numpy as np

from model.decision_tree import ClassificationDT


class ClassificationRandomForest:

    def __init__(self, nb_trees, max_depth, min_entropy, min_elem, max_nb_dim_to_check, max_nb_thresholds, nb_classes):
        self.nb_trees = nb_trees
        self.max_depth = max_depth
        self.min_entropy = min_entropy
        self.min_elem = min_elem
        self.max_nb_dim_to_check = max_nb_dim_to_check
        self.max_nb_thresholds = max_nb_thresholds
        self.nb_classes = nb_classes

    def train(self, inputs, targets):
        self.trees = []
        for i in range(self.nb_trees):
            tree = ClassificationDT(self.max_depth, self.nb_classes, min_elem=self.min_elem, min_metric=self.min_entropy)
            tree.train(inputs, targets, (self.max_nb_dim_to_check, self.max_nb_thresholds))
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
