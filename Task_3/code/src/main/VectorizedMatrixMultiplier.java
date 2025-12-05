package main;

/**
 * Vectorized matrix multiplication implementation.
 * Optimizes for cache locality and simulates SIMD-like operations.
 * 
 * Block Size Impact:
 * - Small blocks (16-32): Better for small matrices, less cache pressure
 * - Medium blocks (64-128): Good balance for most cases, fits L1/L2 cache
 * - Large blocks (256+): Better for very large matrices, may exceed cache
 * 
 * Optimal block size depends on:
 * - CPU cache sizes (L1: ~32KB, L2: ~256KB, L3: ~8MB)
 * - Matrix size and data type
 * - Memory bandwidth and latency
 */
public class VectorizedMatrixMultiplier implements MatrixMultiplier {
    
    private final int blockSize;
    
    /**
     * Creates a vectorized multiplier with specified block size.
     * 
     * @param blockSize Size of blocks for cache optimization (default: 64)
     */
    public VectorizedMatrixMultiplier(int blockSize) {
        this.blockSize = blockSize > 0 ? blockSize : 64;
    }
    
    /**
     * Default constructor with optimal block size.
     */
    public VectorizedMatrixMultiplier() {
        this(64);
    }
    
    @Override
    public double[][] multiply(double[][] A, double[][] B) {
        MatrixUtils.validateMultiplication(A, B);
        
        int rowsA = A.length;
        int colsA = A[0].length;
        int colsB = B[0].length;
        
        double[][] C = MatrixUtils.createResultMatrix(rowsA, colsB);
        
        // Block-wise multiplication for better cache locality
        for (int ii = 0; ii < rowsA; ii += blockSize) {
            for (int jj = 0; jj < colsB; jj += blockSize) {
                for (int kk = 0; kk < colsA; kk += blockSize) {
                    
                    // Calculate block boundaries
                    int iEnd = Math.min(ii + blockSize, rowsA);
                    int jEnd = Math.min(jj + blockSize, colsB);
                    int kEnd = Math.min(kk + blockSize, colsA);
                    
                    // Multiply the current block
                    multiplyBlock(A, B, C, ii, jj, kk, iEnd, jEnd, kEnd);
                }
            }
        }
        
        return C;
    }
    
    /**
     * Multiplies a block of matrices with vectorized operations.
     * Uses loop unrolling and optimized access patterns.
     */
    private void multiplyBlock(double[][] A, double[][] B, double[][] C,
                              int iStart, int jStart, int kStart,
                              int iEnd, int jEnd, int kEnd) {
        
        for (int i = iStart; i < iEnd; i++) {
            for (int j = jStart; j < jEnd; j += 4) { // Unroll by 4 for vectorization
                
                // Initialize accumulators for vectorized computation
                double sum0 = C[i][j];
                double sum1 = (j + 1 < jEnd) ? C[i][j + 1] : 0;
                double sum2 = (j + 2 < jEnd) ? C[i][j + 2] : 0;
                double sum3 = (j + 3 < jEnd) ? C[i][j + 3] : 0;
                
                // Vectorized inner loop
                for (int k = kStart; k < kEnd; k++) {
                    double aik = A[i][k];
                    
                    // Simulate SIMD operations by processing 4 elements at once
                    sum0 += aik * B[k][j];
                    if (j + 1 < jEnd) sum1 += aik * B[k][j + 1];
                    if (j + 2 < jEnd) sum2 += aik * B[k][j + 2];
                    if (j + 3 < jEnd) sum3 += aik * B[k][j + 3];
                }
                
                // Store results
                C[i][j] = sum0;
                if (j + 1 < jEnd) C[i][j + 1] = sum1;
                if (j + 2 < jEnd) C[i][j + 2] = sum2;
                if (j + 3 < jEnd) C[i][j + 3] = sum3;
            }
        }
    }
    
    @Override
    public String getAlgorithmName() {
        return "Vectorized (block size: " + blockSize + ")";
    }
    
    /**
     * Gets the block size used for cache optimization.
     * 
     * @return Block size
     */
    public int getBlockSize() {
        return blockSize;
    }
}