package main;

import java.util.Random;

/**
 * Utility class providing common matrix operations and helper functions.
 */
public class MatrixUtils {
    
    private static final Random random = new Random();
    
    /**
     * Creates a random matrix with specified dimensions.
     * 
     * @param rows Number of rows
     * @param cols Number of columns
     * @return Random matrix filled with values between 0 and 1
     */
    public static double[][] createRandomMatrix(int rows, int cols) {
        double[][] matrix = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = random.nextDouble();
            }
        }
        return matrix;
    }
    
    /**
     * Validates if two matrices can be multiplied.
     * 
     * @param A First matrix
     * @param B Second matrix
     * @throws IllegalArgumentException if matrices cannot be multiplied
     */
    public static void validateMultiplication(double[][] A, double[][] B) {
        if (A == null || B == null) {
            throw new IllegalArgumentException("Matrices cannot be null");
        }
        if (A.length == 0 || B.length == 0) {
            throw new IllegalArgumentException("Matrices cannot be empty");
        }
        if (A[0].length != B.length) {
            throw new IllegalArgumentException(
                String.format("Cannot multiply matrices: A cols (%d) != B rows (%d)", 
                A[0].length, B.length));
        }
    }
    
    /**
     * Creates a zero-filled result matrix with appropriate dimensions.
     * 
     * @param rowsA Number of rows in first matrix
     * @param colsB Number of columns in second matrix
     * @return Zero-filled result matrix
     */
    public static double[][] createResultMatrix(int rowsA, int colsB) {
        return new double[rowsA][colsB];
    }
    
    /**
     * Compares two matrices for equality within a small tolerance.
     * 
     * @param A First matrix
     * @param B Second matrix
     * @param tolerance Maximum allowed difference between elements
     * @return true if matrices are equal within tolerance
     */
    public static boolean matricesEqual(double[][] A, double[][] B, double tolerance) {
        if (A.length != B.length || A[0].length != B[0].length) {
            return false;
        }
        
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < A[0].length; j++) {
                if (Math.abs(A[i][j] - B[i][j]) > tolerance) {
                    return false;
                }
            }
        }
        return true;
    }
    
    /**
     * Prints matrix dimensions and a sample of elements.
     * 
     * @param matrix Matrix to print
     * @param name Name for identification
     */
    public static void printMatrixInfo(double[][] matrix, String name) {
        System.out.printf("%s: %dx%d matrix\n", name, matrix.length, matrix[0].length);
        if (matrix.length <= 5 && matrix[0].length <= 5) {
            for (int i = 0; i < matrix.length; i++) {
                for (int j = 0; j < matrix[0].length; j++) {
                    System.out.printf("%.3f ", matrix[i][j]);
                }
                System.out.println();
            }
        } else {
            System.out.printf("Sample element [0][0] = %.3f\n", matrix[0][0]);
        }
        System.out.println();
    }
}