# -*- coding: utf-8 -*-
"""
Created on Wed May 21 23:36:44 2025

@author: fabia
"""

import numpy as np


class SumTree(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0 # Nuevo atributo para rastrear el número de entradas
        self.data_ptr = 0

    def add(self, priority, data):
        tree_index = self.capacity - 1 + self.data_ptr
        self.data[self.data_ptr] = data
        self.update(tree_index, priority)
        self.data_ptr = (self.data_ptr + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        parent_index = 0
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index
        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]

    def clear(self):
        self.tree[:] = 0
        self.data[:] = 0
        self.data_ptr = 0
        self.n_entries = 0

    def __len__(self):
        return self.n_entries

