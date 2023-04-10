import abc
from abc import abstractmethod, ABC

import numpy as np


class Node:
    def __init__(self):
        self.right_child = None
        self.left_child = None
        self.split_ind = None
        self.split_val = None
        self.terminal_node = None


class DT(ABC):

    def __init__(self, max_depth, min_entropy=0, min_elem=0):
        self.max_depth = max_depth
        self.min_entropy = min_entropy
        self.min_elem = min_elem
        self.root = Node()

    def train(self, inputs, targets):
        metric = self.__calc_metrics(targets)
        self.__nb_dim = inputs.shape[1]
        self.__all_dim = np.arange(self.__nb_dim)

        self.__get_axis, self.__get_threshold = self.__get_all_axis, self.__generate_all_threshold
        self.__build_tree(inputs, targets, self.root, 1, metric)

    def __get_random_axis(self):
        pass

    def __get_all_axis(self):
        pass

    def __create_term_arr(self, target):
        """
        :param target: классы элементов обучающей выборки, дошедшие до узла
        :return: среднее значение
        np.mean(target)
        """
        pass

    def __generate_all_threshold(self, inputs):
        """
        :param inputs: все элементы обучающей выборки выбранной оси
        :return: все пороги, количество порогов определяется значением параметра self.max_nb_thresholds
        Использовать np.min(inputs) и np.max(inputs)
        """
        pass

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

    def __build_splitting_node(self, inputs, targets, entropy, N):
        pass

    def __build_tree(self, inputs, targets, node, depth, metric):

        N = len(targets)
        if depth >= self.max_depth or metric <= self.min_entropy or N <= self.min_elem:
            node.terminal_node = self.__create_term_arr(targets)
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
        return RegressionDT.__disp(targets)

    @staticmethod
    def __disp(targets: np.ndarray) -> float:
        """
        :param targets: классы элементов обучающей выборки, дошедшие до узла
        :return: дисперсия
        np.std(arr)
        """
        pass


class ClassificationDT(DT):

    @staticmethod
    def __calc_metrics(targets: np.ndarray, *args, **kwargs) -> float:
        return ClassificationDT.__shannon_entropy(targets)

    @staticmethod
    def __shannon_entropy(targets) -> float:
        """
                :param targets: классы элементов обучающей выборки, дошедшие до узла

                :return: энтропия
                np.std(arr)
        """
        pass

