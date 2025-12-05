package benchmark;

import main.*;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.ThreadMXBean;
import java.util.ArrayList;
import java.util.List;

/**
 * Utility class for running and analyzing matrix multiplication benchmarks.
 * All benchmarks use 5 warmup runs + 5 measurement runs for statistical reliability.
 */
public class BenchmarkRunner {
    
    /** Number of warmup iterations to stabilize JVM performance */
    private static final int WARMUP_RUNS = 5;
    
    /** Number of measurement iterations for statistical averaging */
    private static final int MEASURE_RUNS = 5;
    
    /**
     * Runs a benchmark for a specific algorithm and matrix size.
     * 
     * @param multiplier The matrix multiplication algorithm to test
     * @param size Size of square matrices to multiply
     * @param warmupRuns Number of warmup runs before measurement
     * @param measureRuns Number of measurement runs to average
     * @return BenchmarkResult containing performance metrics
     */
    public static BenchmarkResult runBenchmark(MatrixMultiplier multiplier, int size,
                                              int warmupRuns, int measureRuns) {
        
        System.out.printf("Benchmarking %s with %dx%d matrices...\n", 
                         multiplier.getAlgorithmName(), size, size);
        
        // Create test matrices
        double[][] A = MatrixUtils.createRandomMatrix(size, size);
        double[][] B = MatrixUtils.createRandomMatrix(size, size);
        
        // Warmup runs
        for (int i = 0; i < warmupRuns; i++) {
            multiplier.multiply(A, B);
        }
        
        // Force garbage collection before measurement
        System.gc();
        Thread.yield();
        
        // Simple real memory measurement
        Runtime runtime = Runtime.getRuntime();
        System.gc(); // Clean slate
        try { Thread.sleep(10); } catch (InterruptedException e) { /* ignore */ }
        long memoryBefore = runtime.totalMemory() - runtime.freeMemory();
        
        // Memory monitoring setup
        long wallClockBefore = System.nanoTime();
        
        // Real memory measurement during actual operations
        long totalTime = 0;
        long peakMemoryUsed = memoryBefore;
        
        for (int i = 0; i < measureRuns; i++) {
            long startTime = System.nanoTime();
            double[][] result = multiplier.multiply(A, B);
            long endTime = System.nanoTime();
            
            // Measure real memory while result is in scope
            long currentMemory = runtime.totalMemory() - runtime.freeMemory();
            peakMemoryUsed = Math.max(peakMemoryUsed, currentMemory);
            
            totalTime += (endTime - startTime);
            
            // Prevent optimization but don't null yet
            if (result[0][0] < -1000) System.out.println("Unexpected");
        }
        
        // Use real measured memory usage
        long memoryUsed = Math.max(0, peakMemoryUsed - memoryBefore);
        long avgTime = totalTime / measureRuns;
        
        // Calculate CPU usage
        long wallClockAfter = System.nanoTime();
        long wallClockElapsed = wallClockAfter - wallClockBefore;

        
        int numThreads = getActualThreadCount(multiplier);
        
        BenchmarkResult result = new BenchmarkResult(
            multiplier.getAlgorithmName(), size, avgTime, numThreads, memoryUsed);
        
        System.out.printf("  Completed: %.2f ms average (%d runs)\n", 
                         result.getExecutionTimeMillis(), measureRuns);
        return result;
    }
    
    private static int getActualThreadCount(MatrixMultiplier multiplier) {
        if (multiplier instanceof ParallelMatrixMultiplier) {
            return ((ParallelMatrixMultiplier) multiplier).getNumThreads();
        } else if (multiplier instanceof AdvancedParallelMatrixMultiplier) {
            return ((AdvancedParallelMatrixMultiplier) multiplier).getNumThreads();
        } else if (multiplier instanceof ForkJoinMatrixMultiplier) {
            return ((ForkJoinMatrixMultiplier) multiplier).getNumThreads();
        }
        return 1;
    }
    
    /**
     * Runs benchmarks for multiple algorithms and analyzes results.
     * 
     * @param multipliers List of algorithms to benchmark
     * @param size Matrix size to test
     * @return List of benchmark results
     */
    public static List<BenchmarkResult> runComparativeBenchmark(
            List<MatrixMultiplier> multipliers, int size) {
        
        List<BenchmarkResult> results = new ArrayList<>();
        
        for (MatrixMultiplier multiplier : multipliers) {
            BenchmarkResult result = runBenchmark(multiplier, size, WARMUP_RUNS, MEASURE_RUNS);
            results.add(result);
        }
        
        return results;
    }
    
    /**
     * Analyzes and prints performance comparison results with comprehensive metrics.
     * 
     * @param results List of benchmark results to analyze
     */
    public static void analyzeResults(List<BenchmarkResult> results) {
        if (results.isEmpty()) return;
        
        // Find baseline (typically the first sequential algorithm)
        BenchmarkResult baseline = results.get(0);
        
        System.out.println("\n=== COMPREHENSIVE PERFORMANCE ANALYSIS ===");
        System.out.println("Baseline: " + baseline.getAlgorithmName());
        System.out.printf("Matrix Size: %dx%d | Baseline Time: %.2f ms\n", 
                baseline.getMatrixSize(), baseline.getMatrixSize(), baseline.getExecutionTimeMillis());
        System.out.println();
        
        // Main results table
        System.out.printf("%-30s %10s %10s %10s %12s %10s\n",
                "Algorithm", "Time (ms)", "Speedup", "Efficiency", "Cores Used", "Memory (MB)");
        System.out.println("-".repeat(95));
        
        for (BenchmarkResult result : results) {
            double speedup = result.getSpeedup(baseline);
            double efficiency = result.getEfficiency(baseline);
            
            System.out.printf("%-30s %10.2f %10.2fx %9.2f%% %12d %10.2f\n",
                    result.getAlgorithmName(),
                    result.getExecutionTimeMillis(),
                    speedup,
                    efficiency * 100,
                    result.getNumThreads(),
                    result.getMemoryUsedMB());
        }
        
        // Detailed efficiency analysis
        analyzeEfficiencyMetrics(results, baseline);
        
        // Resource usage analysis
        analyzeResourceUsage(results, baseline);
        
        System.out.println();
    }
    
    /**
     * Analyzes parallel efficiency metrics in detail.
     */
    private static void analyzeEfficiencyMetrics(List<BenchmarkResult> results, BenchmarkResult baseline) {
        System.out.println("\n--- PARALLEL EFFICIENCY ANALYSIS ---");
        System.out.printf("%-30s %10s %12s %12s %15s\n", 
                "Algorithm", "Threads", "Speedup", "Efficiency", "Speedup/Thread");
        System.out.println("-".repeat(85));
        
        for (BenchmarkResult result : results) {
            if (result.getNumThreads() > 1) {
                double speedup = result.getSpeedup(baseline);
                double efficiency = result.getEfficiency(baseline);
                double speedupPerThread = speedup / result.getNumThreads();
                
                String efficiencyRating = getEfficiencyRating(efficiency);
                
                System.out.printf("%-30s %10d %10.2fx %10.2f%% %13.3f %s\n",
                        result.getAlgorithmName(),
                        result.getNumThreads(),
                        speedup,
                        efficiency * 100,
                        speedupPerThread,
                        efficiencyRating);
            }
        }
    }
    
    /**
     * Analyzes resource usage patterns.
     */
    private static void analyzeResourceUsage(List<BenchmarkResult> results, BenchmarkResult baseline) {
        System.out.println("\n--- RESOURCE USAGE ANALYSIS ---");
        
        int totalCores = Runtime.getRuntime().availableProcessors();
        long totalMemoryMB = Runtime.getRuntime().maxMemory() / (1024 * 1024);
        
        System.out.printf("System Resources: %d cores, %d MB max memory\n", totalCores, totalMemoryMB);
        System.out.println();
        
        System.out.printf("%-30s %12s %12s %15s %12s\n",
                "Algorithm", "Core Usage", "Memory Used", "Memory/Core", "Efficiency");
        System.out.println("-".repeat(90));
        
        for (BenchmarkResult result : results) {
            // Cap core usage at 100% since you cannot use more than 100% of available cores
            double coreUsagePercent = Math.min(100.0, (double) result.getNumThreads() / totalCores * 100);
            double memoryPercent = result.getMemoryUsedMB() / totalMemoryMB * 100;
            double memoryPerCore = result.getMemoryUsedMB() / Math.max(1, result.getNumThreads());
            double efficiency = result.getEfficiency(baseline);
            
            System.out.printf("%-30s %10.1f%% %10.2f%% %13.2f MB %10.2f%%\n",
                    result.getAlgorithmName(),
                    coreUsagePercent,
                    memoryPercent,
                    memoryPerCore,
                    efficiency * 100);
        }
        
        // Find best efficiency
        BenchmarkResult bestEfficiency = results.stream()
            .filter(r -> r.getNumThreads() > 1)
            .max((r1, r2) -> Double.compare(r1.getEfficiency(baseline), r2.getEfficiency(baseline)))
            .orElse(null);
            
        if (bestEfficiency != null) {
            System.out.printf("\n🏆 Best Parallel Efficiency: %s (%.2f%% with %d cores)\n",
                    bestEfficiency.getAlgorithmName(),
                    bestEfficiency.getEfficiency(baseline) * 100,
                    bestEfficiency.getNumThreads());
        }
        
        // Export results to CSV
        CSVExporter.exportResults(results, baseline.getMatrixSize(), "comparison");
    }
    
    /**
     * Gets efficiency rating based on percentage.
     */
    private static String getEfficiencyRating(double efficiency) {
        double percent = efficiency * 100;
        if (percent >= 90) return "⭐⭐⭐ EXCELLENT";
        if (percent >= 70) return "⭐⭐ GOOD";
        if (percent >= 50) return "⭐ FAIR";
        return "❌ POOR";
    }
    
    /**
     * Analyzes speedup patterns and scaling behavior.
     */
    public static void analyzeSpeedupPatterns(List<BenchmarkResult> results) {
        BenchmarkResult baseline = results.get(0);
        
        System.out.println("\n=== SPEEDUP ANALYSIS ===");
        System.out.println("Speedup Formula: T_sequential / T_parallel");
        System.out.println("Efficiency Formula: Speedup / Number_of_Threads");
        System.out.println();
        
        // Group by algorithm type
        System.out.println("Speedup Breakdown:");
        for (BenchmarkResult result : results) {
            double speedup = result.getSpeedup(baseline);
            double timeReduction = (1.0 - (double)result.getExecutionTimeNanos() / baseline.getExecutionTimeNanos()) * 100;
            
            System.out.printf("• %-25s: %.2fx speedup (%.1f%% time reduction)\n",
                    result.getAlgorithmName(), speedup, timeReduction);
            
            if (result.getNumThreads() > 1) {
                double theoreticalMax = result.getNumThreads();
                double achievedPercent = (speedup / theoreticalMax) * 100;
                System.out.printf("  └─ Achieved %.1f%% of theoretical maximum (%.2fx)\n", 
                        achievedPercent, theoreticalMax);
            }
        }
    }
    
    /**
     * Tests correctness of different algorithms by comparing results.
     * 
     * @param multipliers List of algorithms to verify
     * @param size Small matrix size for verification
     */
    public static void verifyCorrectness(List<MatrixMultiplier> multipliers, int size) {
        System.out.println("=== CORRECTNESS VERIFICATION ===");
        
        if (multipliers.isEmpty()) return;
        
        // Create small test matrices
        double[][] A = MatrixUtils.createRandomMatrix(size, size);
        double[][] B = MatrixUtils.createRandomMatrix(size, size);
        
        // Use first algorithm as reference
        double[][] reference = multipliers.get(0).multiply(A, B);
        System.out.printf("Reference: %s\n", multipliers.get(0).getAlgorithmName());
        
        // Compare all other algorithms
        for (int i = 1; i < multipliers.size(); i++) {
            MatrixMultiplier multiplier = multipliers.get(i);
            double[][] result = multiplier.multiply(A, B);
            
            boolean correct = MatrixUtils.matricesEqual(reference, result, 1e-10);
            System.out.printf("%-25s: %s\n", 
                    multiplier.getAlgorithmName(), 
                    correct ? "✓ CORRECT" : "✗ INCORRECT");
        }
        
        System.out.println();
    }
}