# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    math_utils.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: jhue <jhue@student.42lyon.fr>              +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2026/04/14 14:49:36 by jhue              #+#    #+#              #
#    Updated: 2026/04/14 14:52:04 by jhue             ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np


def one_hot(label, num_classes):
    vec = np.zeros(num_classes, dtype=float)
    vec[int(label)] = 1.0
    return vec


def accuracy(model, X, y):
    correct = 0
    for x, target in zip(X, y):
        pred = model.predict(x)
        if pred == target:
            correct += 1
    return correct / len(X)
