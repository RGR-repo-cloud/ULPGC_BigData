package main;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.Semaphore;
import java.util.concurrent.locks.ReentrantLock;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Advanced parallel matrix multiplication demonstrating multiple Java concurrency features:
 * - ExecutorService for thread pool management
 * - Semaphores for resource limiting
 * - Synchronized blocks for critical sections
 * - Parallel streams for functional parallelism
 * - ReentrantLock for fine-grained synchronization
 */
public class AdvancedParallelMatrixMultiplier implements MatrixMultiplier {
    
    private final int numThreads;
    private final boolean useStreams;
    private final boolean useSemaphore;
    private final int maxConcurrentTasks;
    
    // Synchronization objects
    private final Object progressLock = new Object();
    private final ReentrantLock resultLock = new ReentrantLock();
    private volatile int completedRows = 0;
    
    public AdvancedParallelMatrixMultiplier(int numThreads, boolean useStreams, 
                                          boolean useSemaphore, int maxConcurrentTasks) {
        this.numThreads = numThreads > 0 ? numThreads : Runtime.getRuntime().availableProcessors();
        this.useStreams = useStreams;
        this.useSemaphore = useSemaphore;
        this.maxConcurrentTasks = maxConcurrentTasks;
    }
    
    public AdvancedParallelMatrixMultiplier() {
        this(Runtime.getRuntime().availableProcessors(), false, false, Integer.MAX_VALUE);
    }
    
    @Override
    public double[][] multiply(double[][] A, double[][] B) {
        MatrixUtils.validateMultiplication(A, B);
        
        int rowsA = A.length;
        int colsA = A[0].length;
        int colsB = B[0].length;
        
        double[][] C = MatrixUtils.createResultMatrix(rowsA, colsB);
        
        if (useStreams) {
            return multiplyWithStreams(A, B, C);
        } else {
            return multiplyWithExecutorAndSemaphore(A, B, C);
        }
    }
    
    /**
     * Matrix multiplication using parallel streams (functional approach).
     */
    private double[][] multiplyWithStreams(double[][] A, double[][] B, double[][] C) {
        int rowsA = A.length;
        int colsB = B[0].length;
        int colsA = A[0].length;
        
        // Use parallel streams for row-wise computation only (avoid nested parallelism)
        IntStream.range(0, rowsA)
                .parallel()
                .forEach(i -> {
                    // Process each column sequentially within each row
                    for (int j = 0; j < colsB; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < colsA; k++) {
                            sum += A[i][k] * B[k][j];
                        }
                        C[i][j] = sum; // No synchronization needed - different rows
                    }
                    
                    // Update progress with synchronized block
                    synchronized (progressLock) {
                        completedRows++;
                    }
                });
        
        return C;
    }
    
    /**
     * Matrix multiplication using ExecutorService with Semaphore for resource control.
     */
    private double[][] multiplyWithExecutorAndSemaphore(double[][] A, double[][] B, double[][] C) {
        int rowsA = A.length;
        int colsB = B[0].length;
        int colsA = A[0].length;
        
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        List<Future<?>> futures = new ArrayList<>();
        
        // Semaphore to limit concurrent tasks (demonstrates resource management)
        Semaphore semaphore = useSemaphore ? new Semaphore(maxConcurrentTasks) : null;
        
        try {
            // Create tasks for each row
            for (int i = 0; i < rowsA; i++) {
                final int row = i;
                
                Future<?> future = executor.submit(() -> {
                    try {
                        // Acquire semaphore permit if using semaphore control
                        if (semaphore != null) {
                            semaphore.acquire();
                        }
                        
                        processRow(A, B, C, row, colsB, colsA);
                        
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        throw new RuntimeException("Task interrupted", e);
                    } finally {
                        // Release semaphore permit
                        if (semaphore != null) {
                            semaphore.release();
                        }
                        
                        // Update progress using synchronized block
                        synchronized (progressLock) {
                            completedRows++;
                        }
                    }
                });
                
                futures.add(future);
            }
            
            // Wait for all tasks to complete
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
    
    /**
     * Processes a single row with fine-grained locking.
     */
    private void processRow(double[][] A, double[][] B, double[][] C, 
                           int row, int colsB, int colsA) {
        
        for (int j = 0; j < colsB; j++) {
            double sum = 0.0;
            
            // Compute dot product for C[row][j]
            for (int k = 0; k < colsA; k++) {
                sum += A[row][k] * B[k][j];
            }
            
            // Use ReentrantLock for fine-grained synchronization
            resultLock.lock();
            try {
                C[row][j] = sum;
            } finally {
                resultLock.unlock();
            }
        }
    }
    
    @Override
    public String getAlgorithmName() {
        StringBuilder name = new StringBuilder("Advanced Parallel (");
        name.append(numThreads).append(" threads");
        if (useStreams) {
            name.append(", streams");
        }
        if (useSemaphore) {
            name.append(", semaphore:").append(maxConcurrentTasks);
        }
        name.append(")");
        return name.toString();
    }
    
    /**
     * Gets the number of completed rows (thread-safe).
     */
    public int getCompletedRows() {
        synchronized (progressLock) {
            return completedRows;
        }
    }
    
    /**
     * Resets the progress counter.
     */
    public void resetProgress() {
        synchronized (progressLock) {
            completedRows = 0;
        }
    }
    
    public int getNumThreads() {
        return numThreads;
    }
    
    public boolean isUsingStreams() {
        return useStreams;
    }
    
    public boolean isUsingSemaphore() {
        return useSemaphore;
    }
}