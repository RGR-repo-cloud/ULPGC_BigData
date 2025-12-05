package main;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.ArrayList;
import java.util.List;

/**
 * Parallel matrix multiplication using multiple threads.
 * Implements row-wise parallelization with thread pool management.
 */
public class ParallelMatrixMultiplier implements MatrixMultiplier {
    
    private final int numThreads;
    
    /**
     * Creates a parallel multiplier with specified number of threads.
     * 
     * @param numThreads Number of threads to use (defaults to available processors)
     */
    public ParallelMatrixMultiplier(int numThreads) {
        this.numThreads = numThreads > 0 ? numThreads : Runtime.getRuntime().availableProcessors();
    }
    
    /**
     * Default constructor using all available processors.
     */
    public ParallelMatrixMultiplier() {
        this(Runtime.getRuntime().availableProcessors());
    }
    
    @Override
    public double[][] multiply(double[][] A, double[][] B) {
        MatrixUtils.validateMultiplication(A, B);
        
        int rowsA = A.length;
        int colsA = A[0].length;
        int colsB = B[0].length;
        
        double[][] C = MatrixUtils.createResultMatrix(rowsA, colsB);
        
        // Use thread pool for efficient thread management
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        List<Future<?>> futures = new ArrayList<>();
        
        // Atomic counter for dynamic work distribution
        AtomicInteger rowCounter = new AtomicInteger(0);
        
        // Create worker tasks
        for (int t = 0; t < numThreads; t++) {
            Future<?> future = executor.submit(() -> {
                int row;
                // Dynamic work distribution - each thread takes next available row
                while ((row = rowCounter.getAndIncrement()) < rowsA) {
                    for (int j = 0; j < colsB; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < colsA; k++) {
                            sum += A[row][k] * B[k][j];
                        }
                        // No synchronization needed - each thread writes to different rows
                        C[row][j] = sum;
                    }
                }
            });
            futures.add(future);
        }
        
        // Wait for all tasks to complete
        try {
            for (Future<?> future : futures) {
                future.get();
            }
        } catch (Exception e) {
            throw new RuntimeException("Parallel execution failed", e);
        } finally {
            executor.shutdown();
        }
        
        return C;
    }
    
    @Override
    public String getAlgorithmName() {
        return "Parallel (" + numThreads + " threads)";
    }
    
    /**
     * Gets the number of threads used by this multiplier.
     * 
     * @return Number of threads
     */
    public int getNumThreads() {
        return numThreads;
    }
}