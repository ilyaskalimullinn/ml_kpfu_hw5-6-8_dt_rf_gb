import numpy as np


class AdaBoost:
    def __init__(self, M):
        self.M = M

    def __init_weights(self, N):
        """ initialisation of input variables weights"""
        pass

    def update_weights(self, gt, predict, weights, weight_weak_classifiers):
        """ update weights functions DO NOT use loops"""
        pass

    def calculate_error(self, gt, predict, weights):
        """ weak classifier error calculation DO NOT use loops"""
        pass

    def calculate_classifier_weight(self, gt, predict, weights):
        """ weak classifier weight calculation DO NOT use loops"""
        pass

    def train(self, target, vectors):
        """ train model"""
        pass

    def get_prediction(self, vectors):
        """ adaboost get prediction """
        pass
