package benchmark;

import main.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Main class for running comprehensive matrix multiplication benchmarks.
 * Tests different algorithms with various matrix sizes and analyzes performance.
 */
public class PerformanceBenchmark {
    
    public static void main(String[] args) {
        System.out.println("=== PARALLEL MATRIX MULTIPLICATION BENCHMARK ===");
        System.out.println("Big Data Course - Task 3");
        System.out.println("Testing parallel computing techniques for matrix multiplication");
        System.out.println();
        
        // Clear existing CSV files to start fresh
        CSVExporter.clearExistingResults();
        
        // Print system information
        printSystemInfo();
        
        // Create different multiplication algorithms
        List<MatrixMultiplier> algorithms = createAlgorithms();
        
        // Verify correctness with small matrices
        BenchmarkRunner.verifyCorrectness(algorithms, 10);
        
        // Phase 1: Hyperparameter Optimization on 512x512 matrices
        System.out.println("\n" + "=".repeat(80));
        System.out.println("PHASE 1: HYPERPARAMETER OPTIMIZATION (512x512 matrices)");
        System.out.println("=".repeat(80));
        
        HyperparameterResults optimalParams = optimizeHyperparameters();
        
        // Phase 2: Algorithm Comparison with Optimal Parameters
        System.out.println("\n" + "=".repeat(80));
        System.out.println("PHASE 2: ALGORITHM COMPARISON WITH OPTIMAL PARAMETERS");
        System.out.println("=".repeat(80));
        
        int[] testSizes = {128, 256, 512, 1024};
        for (int size : testSizes) {
            compareAlgorithmsOptimal(size, optimalParams);
        }
        
        // Phase 3: Concurrency Mechanisms with Optimal Parameters
        System.out.println("\n" + "=".repeat(80));
        System.out.println("PHASE 3: CONCURRENCY MECHANISMS WITH OPTIMAL PARAMETERS");
        System.out.println("=".repeat(80));
        
        for (int size : testSizes) {
            compareConcurrencyMechanismsOptimal(size, optimalParams);
        }
        
        System.out.println("\n" + "=".repeat(60));
        System.out.println("BENCHMARK COMPLETED");
        System.out.println("=".repeat(60));
        System.out.println("\n🎯 Key Insights:");
        System.out.println("  • Hyperparameter optimization critical for performance");
        System.out.println("\n• Vectorized algorithms excel with proper block tuning");
        System.out.println("  • Parallel efficiency varies significantly with configuration");
        System.out.println("  • Java concurrency features provide diverse optimization strategies");
        
        // Export comprehensive benchmark summary
        CSVExporter.exportBenchmarkSummary("Complete hyperparameter optimization and comparison benchmark");
        
        System.out.println("\n📁 All benchmark results exported to CSV files in 'results/' directory");
        System.out.println("💡 To generate plots manually, run: python3 plotting.py");
    }
    
    /**
     * Creates a basic list of algorithms for initial testing (unused in new structure).
     */
    private static List<MatrixMultiplier> createAlgorithms() {
        List<MatrixMultiplier> algorithms = new ArrayList<>();
        algorithms.add(new BasicMatrixMultiplier());
        return algorithms;
    }
    
    /**
     * Results container for optimal hyperparameters.
     */
    private static class HyperparameterResults {
        int optimalBlockSize;
        int optimalThreads;
        int optimalSemaphorePermits;
        int optimalForkJoinThreshold;
        
        HyperparameterResults(int blockSize, int threads, int semaphorePermits, int forkJoinThreshold) {
            this.optimalBlockSize = blockSize;
            this.optimalThreads = threads;
            this.optimalSemaphorePermits = semaphorePermits;
            this.optimalForkJoinThreshold = forkJoinThreshold;
        }
    }
    
    /**
     * Comprehensive hyperparameter optimization on 512x512 matrices.
     */
    private static HyperparameterResults optimizeHyperparameters() {
        int testSize = 512;
        
        // 1. Optimize Block Size for Vectorization
        int optimalBlockSize = optimizeBlockSize(testSize);
        
        // 2. Optimize Thread Count for Parallel Execution
        int optimalThreads = optimizeThreadCount(testSize);
        
        // 3. Optimize Semaphore Permits
        int optimalSemaphorePermits = optimizeSemaphorePermits(testSize, optimalThreads);
        
        // 4. Optimize Fork-Join Threshold
        int optimalForkJoinThreshold = optimizeForkJoinThreshold(testSize, optimalThreads);
        
        HyperparameterResults results = new HyperparameterResults(
            optimalBlockSize, optimalThreads, optimalSemaphorePermits, optimalForkJoinThreshold);
        
        System.out.println("\n🎯 OPTIMAL HYPERPARAMETERS FOUND:");
        System.out.println("-".repeat(50));
        System.out.printf("• Block Size (Vectorized): %d\n", optimalBlockSize);
        System.out.printf("• Thread Count (Parallel): %d\n", optimalThreads);
        System.out.printf("• Semaphore Permits: %d\n", optimalSemaphorePermits);
        System.out.printf("• Fork-Join Threshold: %d\n", optimalForkJoinThreshold);
        System.out.println("-".repeat(50));
        
        return results;
    }
    
    /**
     * Optimize block size for vectorized matrix multiplication.
     */
    private static int optimizeBlockSize(int matrixSize) {
        System.out.println("\n--- OPTIMIZING BLOCK SIZE ---");
        
        int[] blockSizes = {16, 32, 64, 128, 256, 512};
        List<MatrixMultiplier> blockTests = new ArrayList<>();
        
        for (int blockSize : blockSizes) {
            blockTests.add(new VectorizedMatrixMultiplier(blockSize));
        }
        
        return findOptimalParameter(blockTests, blockSizes, matrixSize, "Block Size", "block size");
    }
    
    /**
     * Optimize thread count for parallel execution.
     */
    private static int optimizeThreadCount(int matrixSize) {
        System.out.println("\n--- OPTIMIZING THREAD COUNT ---");
        
        int maxThreads = Runtime.getRuntime().availableProcessors();
        int[] threadCounts = {1, 2, 4, 6, 8, 10, 12, 16};
        List<MatrixMultiplier> threadTests = new ArrayList<>();
        List<Integer> validCounts = new ArrayList<>();
        
        // Limit to physical cores for CPU-intensive matrix multiplication
        // Oversubscription typically hurts performance for compute-bound tasks
        for (int threads : threadCounts) {
            if (threads <= maxThreads) {
                threadTests.add(new ParallelMatrixMultiplier(threads));
                validCounts.add(threads);
            }
        }
        
        return findOptimalParameter(threadTests, validCounts.stream().mapToInt(i -> i).toArray(), 
                                  matrixSize, "Threads", "thread count");
    }
    
    /**
     * Optimize semaphore permits for resource limiting.
     */
    private static int optimizeSemaphorePermits(int matrixSize, int threads) {
        System.out.println("\n--- OPTIMIZING SEMAPHORE PERMITS ---");
        
        int[] permits = {1, 2, 4, 8, 16};
        List<MatrixMultiplier> semaphoreTests = new ArrayList<>();
        
        for (int permit : permits) {
            semaphoreTests.add(new AdvancedParallelMatrixMultiplier(threads, false, true, permit));
        }
        
        return findOptimalParameter(semaphoreTests, permits, matrixSize, "Permits", "semaphore permits");
    }
    
    /**
     * Optimize Fork-Join threshold.
     */
    private static int optimizeForkJoinThreshold(int matrixSize, int parallelism) {
        System.out.println("\n--- OPTIMIZING FORK-JOIN THRESHOLD ---");
        
        int[] thresholds = {16, 32, 64, 128, 256, 512};
        List<MatrixMultiplier> forkJoinTests = new ArrayList<>();
        
        for (int threshold : thresholds) {
            forkJoinTests.add(new ForkJoinMatrixMultiplier(threshold, parallelism, parallelism));
        }
        
        return findOptimalParameter(forkJoinTests, thresholds, matrixSize, "Threshold", "Fork-Join threshold");
    }
    
    /**
     * Helper method to find optimal parameter from benchmark results.
     */
    private static int findOptimalParameter(List<MatrixMultiplier> algorithms, int[] parameters, 
                                          int matrixSize, String columnName, String parameterName) {
        List<BenchmarkResult> results = BenchmarkRunner.runComparativeBenchmark(algorithms, matrixSize);
        
        System.out.printf("%-15s %10s\n", columnName, "Time (ms)");
        System.out.println("-".repeat(30));
        
        BenchmarkResult best = results.get(0);
        int optimal = parameters[0];
        
        for (int i = 0; i < results.size(); i++) {
            BenchmarkResult result = results.get(i);
            System.out.printf("%-15d %10.2f\n", parameters[i], result.getExecutionTimeMillis());
            
            if (result.getExecutionTimeNanos() < best.getExecutionTimeNanos()) {
                best = result;
                optimal = parameters[i];
            }
        }
        
        System.out.printf("\n✓ Optimal %s: %d (%.2f ms)\n", parameterName, optimal, best.getExecutionTimeMillis());
        
        // Export ONLY to hyperparameter optimization file (not main performance files)
        CSVExporter.exportHyperparameterResults(results, parameterName, matrixSize);
        
        return optimal;
    }
    
    /**
     * Compare basic vs vectorized vs parallel algorithms with optimal parameters.
     */
    private static void compareAlgorithmsOptimal(int matrixSize, HyperparameterResults params) {
        System.out.printf("\n=== ALGORITHM COMPARISON %dx%d (OPTIMAL PARAMS) ===\n", matrixSize, matrixSize);
        
        List<MatrixMultiplier> algorithms = new ArrayList<>();
        algorithms.add(new BasicMatrixMultiplier());
        algorithms.add(new VectorizedMatrixMultiplier(params.optimalBlockSize));
        algorithms.add(new ParallelMatrixMultiplier(params.optimalThreads));
        
        List<BenchmarkResult> results = BenchmarkRunner.runComparativeBenchmark(algorithms, matrixSize);
        BenchmarkRunner.analyzeResults(results);
        
        // Export ONLY optimal algorithm comparison results (not hyperparameter exploration)
        CSVExporter.exportOptimalResults(results, matrixSize, "algorithm_comparison");
    }
    
    /**
     * Compare different concurrency mechanisms with optimal parameters.
     */
    private static void compareConcurrencyMechanismsOptimal(int matrixSize, HyperparameterResults params) {
        System.out.printf("\n=== CONCURRENCY MECHANISMS %dx%d (OPTIMAL PARAMS) ===\n", matrixSize, matrixSize);
        System.out.println("Using optimal parameters:");
        System.out.printf("  • Threads: %d, Semaphore: %d, Fork-Join Threshold: %d\n", 
            params.optimalThreads, params.optimalSemaphorePermits, params.optimalForkJoinThreshold);
        System.out.println();
        
        List<MatrixMultiplier> concurrencyTests = new ArrayList<>();
        
        // Baseline
        concurrencyTests.add(new BasicMatrixMultiplier());
        
        // Optimized algorithms
        concurrencyTests.add(new ParallelMatrixMultiplier(params.optimalThreads));
        concurrencyTests.add(new AdvancedParallelMatrixMultiplier(
            params.optimalThreads, false, false, Integer.MAX_VALUE)); // Basic executor
        concurrencyTests.add(new AdvancedParallelMatrixMultiplier(
            params.optimalThreads, false, true, params.optimalSemaphorePermits)); // + Semaphore
        concurrencyTests.add(new AdvancedParallelMatrixMultiplier(
            params.optimalThreads, true, false, Integer.MAX_VALUE)); // + Parallel Streams
        concurrencyTests.add(new ForkJoinMatrixMultiplier(
            params.optimalForkJoinThreshold, params.optimalThreads, params.optimalThreads)); // Fork-Join
        
        List<BenchmarkResult> results = BenchmarkRunner.runComparativeBenchmark(concurrencyTests, matrixSize);
        BenchmarkRunner.analyzeResults(results);
        
        // Export ONLY optimal concurrency mechanisms results (not hyperparameter exploration)
        CSVExporter.exportOptimalResults(results, matrixSize, "concurrency_mechanisms");
        
        System.out.println("\n🛠️ Java Concurrency Features: ExecutorService, Parallel Streams,");
        System.out.println("   Semaphore, Synchronized blocks, AtomicInteger, ReadWriteLock, Fork-Join\n");
    }
    

    
    /**
     * Prints system information for benchmark context.
     */
    private static void printSystemInfo() {
        Runtime runtime = Runtime.getRuntime();
        
        System.out.println("System Information:");
        System.out.printf("  Available Processors: %d\n", runtime.availableProcessors());
        System.out.printf("  Max Memory: %.2f MB\n", runtime.maxMemory() / (1024.0 * 1024.0));
        System.out.printf("  Total Memory: %.2f MB\n", runtime.totalMemory() / (1024.0 * 1024.0));
        System.out.printf("  Java Version: %s\n", System.getProperty("java.version"));
        System.out.printf("  OS: %s %s\n", 
                         System.getProperty("os.name"), 
                         System.getProperty("os.arch"));
        System.out.println();
    }
}