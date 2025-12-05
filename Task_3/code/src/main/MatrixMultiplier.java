package main;

/**
 * Interface defining the contract for matrix multiplication implementations.
 * This allows easy switching between different algorithms for benchmarking.
 */
public interface MatrixMultiplier {
    
    /**
     * Multiplies two matrices A and B to produce result matrix C.
     * 
     * @param A First matrix (m x n)
     * @param B Second matrix (n x p)
     * @return Result matrix C (m x p)
     * @throws IllegalArgumentException if matrices cannot be multiplied
     */
    double[][] multiply(double[][] A, double[][] B);
    
    /**
     * Returns the name of the multiplication algorithm.
     * 
     * @return Algorithm name for identification
     */
    String getAlgorithmName();
}