"""
Statistical Utilities for Python
"""
import statistics


def calculate_statistics(values):
    """
    Calculate comprehensive statistics for a list of values
    """
    return {
        'avg': statistics.mean(values),
        'min': min(values),
        'max': max(values),
        'std_dev': statistics.stdev(values) if len(values) > 1 else 0
    }