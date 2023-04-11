from abc import abstractmethod, ABC
from typing import Union

import numpy as np


class Node:
    def __init__(self):
        self.right_child = None
        self.left_child = None
        self.split_ind = None
        self.split_val = None
        self.terminal_node = None


class DT(ABC):

    def __init__(self, max_depth, min_metric=0, min_elem=1):
        self.max_depth = max_depth
        self.min_metric = min_metric
        self.min_elem = min_elem
        self.root = Node()

    def train(self, inputs, targets):
        metric = self._calc_metrics(targets)
        self.D = inputs.shape[1]
        self.__all_dim = np.arange(self.D)

        self.__get_axis, self.__get_threshold = self.__get_all_axis, self.__generate_all_threshold
        self.__build_tree(inputs, targets, self.root, 1, metric)

    def __get_random_axis(self):
        pass

    def __get_all_axis(self):
        """
        Feature selection function
        :return: all indexes of input array - a range 0...d-1
        """
        return self.__all_dim

    @abstractmethod
    def _create_term_value(self, targets: np.ndarray) -> Union[np.ndarray, float]:
        """
        :param targets: target values of train set that have reached this terminal node
        :return: terminal value (probability array for classification or a float for regression)
        """
        pass

    def __generate_all_threshold(self, inputs: np.ndarray) -> np.ndarray:
        """
        :param inputs: all train inputs (just one feature from all of it, array of shape (N, 1))
        :return: all thresholds - all unique values of this feature
        """
        return np.unique(inputs)

    def __generate_random_threshold(self, inputs):
        """
        :param inputs: все элементы обучающей выборки(дошедшие до узла) выбранной оси
        :return: пороги, выбранные с помощью равномерного распределения.
        Количество порогов определяется значением параметра self.max_nb_thresholds
        """
        pass

    @staticmethod
    @abstractmethod
    def _calc_metrics(targets: np.ndarray, *args, **kwargs) -> float:
        pass

    def __inf_gain(self, targets_left: np.ndarray, targets_right: np.ndarray, parent_metric: Union[float, None] = None,
                   N: Union[int, None] = None):
        """
        :param targets_left: targets for elements that went to left node
        :param targets_right: targets for elements that went to right node
        :param parent_metric: metric value for parent node
        :param N: number of elements that made it to the parent node
        :return: information gain, metric for left node, metric for right node
        """
        if parent_metric is None:
            parent_metric = self._calc_metrics(np.hstack([targets_left, targets_right]))
        if N is None:
            N = targets_left.shape[0] + targets_right.shape[0]

        metric_left = self._calc_metrics(targets_left)
        metric_right = self._calc_metrics(targets_right)

        expected_metric = (targets_left.shape[0] / N) * metric_left + (targets_right.shape[0] / N) * metric_right

        inf_gain = parent_metric - expected_metric

        return inf_gain, metric_left, metric_right

    def __build_splitting_node(self, inputs: np.ndarray, targets: np.ndarray, metric: float, N: Union[int, None] = None):
        """

        :param inputs: train inputs that came to this node
        :param targets: train targets that came to this node
        :param metric: metric (entropy or variance) for this node
        :param N: amount of elements that came to this node
        :return: feature index for feature selection function,
                threshold for splitting function,
                indexes for elements that go to the left child node,
                indexes for elements that go to the right child node,
                metric value for left node,
                metric value for right node
        """
        if N is None:
            N = len(targets)

        information_gain_max = 0
        idx_right_best = None
        idx_left_best = None
        metric_left_best = None
        metric_right_best = None
        ax_best = None
        th_best = None


        for ax in self.__get_axis():
            for th in self.__get_threshold(inputs[:, ax]):
                idx_right = np.where(inputs[:, ax] > th, True, False)
                idx_left = ~idx_right

                information_gain, metric_left, metric_right = self.__inf_gain(targets[idx_left], targets[idx_right], N=N)

                if information_gain >= information_gain_max:
                    ax_best = ax
                    th_best = th
                    information_gain_max = information_gain
                    idx_right_best = idx_right
                    idx_left_best = idx_left
                    metric_left_best = metric_left
                    metric_right_best = metric_right

        return ax_best, th_best, idx_left_best, idx_right_best, metric_left_best, metric_right_best,

    def __build_tree(self, inputs, targets, node, depth, metric):

        N = len(targets)
        if depth >= self.max_depth or metric <= self.min_metric or N <= self.min_elem:
            node.terminal_node = self._create_term_value(targets)
        else:

            ax_max, th_max, ind_left_max, ind_right_max, disp_left_max, disp_right_max = self.__build_splitting_node(
                inputs, targets, metric, N)
            node.split_ind = ax_max
            node.split_val = th_max
            node.left_child = Node()
            node.right_child = Node()
            self.__build_tree(inputs[ind_left_max], targets[ind_left_max], node.left_child, depth + 1, disp_left_max)
            self.__build_tree(inputs[ind_right_max], targets[ind_right_max], node.right_child, depth + 1, disp_right_max)

    def get_predictions(self, inputs: np.ndarray) -> np.ndarray:
        """
        :param inputs: вектора характеристик
        :return: предсказания целевых значений
        """
        results = []
        for input in inputs:
            node = self.root

            while node.terminal_node is None:
                if input[node.split_ind] > node.split_val:
                    node = node.right_child
                else:
                    node = node.left_child

            results.append(node.terminal_node)

        return np.vstack(results)


class RegressionDT(DT):
    @staticmethod
    def _calc_metrics(targets: np.ndarray, *args, **kwargs) -> float:
        return RegressionDT.__variance(targets)

    def _create_term_value(self, targets: np.ndarray) -> Union[np.ndarray, float]:
        return targets.mean()

    @staticmethod
    def __variance(targets: np.ndarray) -> float:
        """
        :param targets: train targets that made it to current node
        :return: variance (dispersion)
        """
        if len(targets) == 0:
            return 0
        var = float(np.var(targets))
        return var


class ClassificationDT(DT):

    def __init__(self, max_depth, number_classes, min_metric=0, min_elem=1):
        super().__init__(max_depth, min_metric, min_elem)
        self.K = number_classes

    @staticmethod
    def _calc_metrics(targets: np.ndarray, *args, **kwargs) -> float:
        return ClassificationDT.__shannon_entropy(targets)

    def _create_term_value(self, targets: np.ndarray) -> np.ndarray:
        y = np.bincount(targets, minlength=self.K)
        y = y / len(targets)
        return y

    def get_predictions(self, inputs: np.ndarray, return_probability_vector: bool = True) -> np.ndarray:
        predictions = super().get_predictions(inputs)
        if return_probability_vector:
            return predictions
        return predictions.argmax(axis=1)

    @staticmethod
    def __shannon_entropy(targets) -> float:
        """
        :param targets: train targets that made it to current node
        :return: entropy
        """
        p = np.unique(targets, return_counts=True)[1]
        p = p / targets.shape[0]
        res = -np.sum(p * np.log2(p))
        return res
