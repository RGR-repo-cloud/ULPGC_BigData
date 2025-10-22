/*
 * Matrix Utilities for C - Header file
 */
#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

double** create_random_matrix(int size);
double** create_zero_matrix(int size);
void free_matrix(double** matrix, int size);

#endif