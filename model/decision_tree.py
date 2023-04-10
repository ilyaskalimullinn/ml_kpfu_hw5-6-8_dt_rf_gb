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

    def __init__(self, max_depth, min_metric=0, min_elem=0):
        self.max_depth = max_depth
        self.min_metric = min_metric
        self.min_elem = min_elem
        self.root = Node()

    def train(self, inputs, targets):
        metric = self.__calc_metrics(targets)
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
    def __create_term_value(self, targets: np.ndarray) -> Union[np.ndarray, float]:
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
    def __calc_metrics(targets: np.ndarray, *args, **kwargs) -> float:
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
            parent_metric = self.__calc_metrics(np.vstack(targets_left, targets_right))
        if N is None:
            N = targets_left.shape[0] + targets_right.shape[0]

        metric_left = self.__calc_metrics(targets_left)
        metric_right = self.__calc_metrics(targets_right)

        expected_metric = (targets_left.shape[0] / N) * metric_left + (targets_right.shape[0] / N) * metric_right

        return parent_metric - expected_metric, metric_left, metric_right

    def __build_splitting_node(self, inputs, targets, metric, N):
        pass

    def __build_tree(self, inputs, targets, node, depth, metric):

        N = len(targets)
        if depth >= self.max_depth or metric <= self.min_metric or N <= self.min_elem:
            node.terminal_node = self.__create_term_value(targets)
        else:

            ax_max, tay_max, ind_left_max, ind_right_max, disp_left_max, disp_right_max = self.__build_splitting_node(
                inputs, targets, metric, N)
            node.split_ind = ax_max
            node.split_val = tay_max
            node.left = Node()
            node.right = Node()
            self.__build_tree(inputs[ind_left_max], targets[ind_left_max], node.left, depth + 1, disp_left_max)
            self.__build_tree(inputs[ind_right_max], targets[ind_right_max], node.right, depth + 1, disp_right_max)

    def get_predictions(self, inputs):
        """
        :param inputs: вектора характеристик
        :return: предсказания целевых значений
        """
        pass


class RegressionDT(DT):
    @staticmethod
    def __calc_metrics(targets: np.ndarray, *args, **kwargs) -> float:
        return RegressionDT.__variance(targets)

    def __create_term_value(self, targets: np.ndarray) -> Union[np.ndarray, float]:
        return targets.mean()

    @staticmethod
    def __variance(targets: np.ndarray) -> float:
        """
        :param targets: train targets that made it to current node
        :return: variance (dispersion)
        """
        return float(np.var(targets))


class ClassificationDT(DT):

    def __init__(self, max_depth, number_classes, min_metric=0, min_elem=0):
        super().__init__(max_depth, min_metric, min_elem)
        self.K = number_classes

    @staticmethod
    def __calc_metrics(targets: np.ndarray, *args, **kwargs) -> float:
        return ClassificationDT.__shannon_entropy(targets)

    def __create_term_value(self, targets: np.ndarray) -> np.ndarray:
        y = np.arange(self.K)
        y[targets] += 1
        y = y / targets.shape[0]
        return y

    @staticmethod
    def __shannon_entropy(targets) -> float:
        """
        :param targets: train targets that made it to current node
        :return: entropy
        """
        p = np.unique(targets, return_counts=True)[1]
        p = p / targets.shape[0]
        print(p)
        return -np.sum(p * np.log2(p))

