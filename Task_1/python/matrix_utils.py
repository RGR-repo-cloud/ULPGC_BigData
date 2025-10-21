"""
Matrix Utilities for Python
"""
import random


def create_random_matrix(size):
    """Create a random matrix of given size"""
    return [[random.random() for _ in range(size)] for _ in range(size)]


def create_zero_matrix(size):
    """Create a zero matrix of given size"""
    return [[0.0 for _ in range(size)] for _ in range(size)]