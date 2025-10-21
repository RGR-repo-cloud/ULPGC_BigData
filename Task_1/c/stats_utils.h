/*
 * Statistical Utilities for C - Header file
 */
#ifndef STATS_UTILS_H
#define STATS_UTILS_H

double calculate_average(double* values, int length);
double find_min(double* values, int length);
double find_max(double* values, int length);
double calculate_std_dev(double* values, int length);

#endif