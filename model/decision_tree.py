import abc
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
    def __create_term_value(self, target: np.ndarray) -> Union[np.ndarray, float]:
        """
        :param target: target values of train set that have reached this terminal node
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

    def __inf_gain(self, targets_left, targets_right, metric, N):
        """
        :param targets_left: targets для элементов попавших в левый узел
        :param targets_right: targets для элементов попавших в правый узел
        :param N: количество элементов, дошедших до узла родителя
        :return: information gain, энтропия для левого узла, энтропия для правого узла
        ТУТ ТОЖЕ НЕ ЦИКЛОВ, используйте собственную фунцию self.__disp
        """
        pass

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

    def __create_term_value(self, target: np.ndarray) -> Union[np.ndarray, float]:
        return target.mean()

    @staticmethod
    def __variance(targets: np.ndarray) -> float:
        """
        :param targets: train targets that made it to current node
        :return: variance (dispersion)
        """
        return float(np.var(targets))


class ClassificationDT(DT):

    @staticmethod
    def __calc_metrics(targets: np.ndarray, *args, **kwargs) -> float:
        return ClassificationDT.__shannon_entropy(targets)

    def __create_term_value(self, target: np.ndarray) -> Union[np.ndarray, float]:
        #  todo
        pass

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

