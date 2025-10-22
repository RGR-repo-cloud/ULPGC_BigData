/*
 * Statistical Utilities for C
 */
#include <math.h>
#include "stats_utils.h"

/**
 * Calculate average of an array
 */
double calculate_average(double* values, int length) {
    double sum = 0;
    for (int i = 0; i < length; i++) {
        sum += values[i];
    }
    return sum / length;
}

/**
 * Find minimum value in array
 */
double find_min(double* values, int length) {
    double min = values[0];
    for (int i = 1; i < length; i++) {
        if (values[i] < min) min = values[i];
    }
    return min;
}

/**
 * Find maximum value in array
 */
double find_max(double* values, int length) {
    double max = values[0];
    for (int i = 1; i < length; i++) {
        if (values[i] > max) max = values[i];
    }
    return max;
}

/**
 * Calculate standard deviation
 */
double calculate_std_dev(double* values, int length) {
    if (length <= 1) return 0;
    
    double mean = calculate_average(values, length);
    double sum_squared_diffs = 0;
    
    for (int i = 0; i < length; i++) {
        double diff = values[i] - mean;
        sum_squared_diffs += diff * diff;
    }
    
    return sqrt(sum_squared_diffs / (length - 1));
}