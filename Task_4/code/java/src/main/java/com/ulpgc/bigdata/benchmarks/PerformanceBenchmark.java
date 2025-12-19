package com.ulpgc.bigdata.benchmarks;

import com.ulpgc.bigdata.production.DistributedMatrixMultiplier;

import java.io.FileWriter;
import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Performance benchmark for matrix multiplication approaches.
 */
public class PerformanceBenchmark {
    
    private static final int NUM_RUNS = 5;
    private static final int[] MATRIX_SIZES = {64, 128, 256, 512, 1024, 2048};
    private static String METHOD = "all";  // all, basic, parallel, distributed
    
    public static void main(String[] args) {
        // Parse command line arguments
        for (int i = 0; i < args.length; i++) {
            if (args[i].equals("--method") && i + 1 < args.length) {
                METHOD = args[i + 1];
            }
        }
        
        System.out.println("============================================================");
        System.out.println("MATRIX MULTIPLICATION PERFORMANCE BENCHMARK (Java)");
        System.out.println("============================================================");
        System.out.println("Configuration:");
        System.out.println("  - Runs per test: " + NUM_RUNS);
        System.out.println("  - Matrix sizes: " + arrayToString(MATRIX_SIZES));
        System.out.println("  - CPU cores: " + Runtime.getRuntime().availableProcessors());
        System.out.println("  - Mode: " + METHOD.substring(0, 1).toUpperCase() + METHOD.substring(1) + 
                          (METHOD.equals("all") ? " methods" : " method"));
        System.out.println("============================================================\n");
        
        DistributedMatrixMultiplier multiplier = new DistributedMatrixMultiplier();
        boolean distributedAvailable = false;
        
        // Only connect to Hazelcast if distributed method is needed
        if (METHOD.equals("all") || METHOD.equals("distributed")) {
            distributedAvailable = multiplier.connect();
            
            if (!distributedAvailable) {
                System.out.println("WARNING: Could not connect to Hazelcast. Distributed tests will be skipped.\n");
            }
        }
        
        List<BenchmarkResult> allResults = new ArrayList<>();
        
        try {
            for (int size : MATRIX_SIZES) {
                System.out.println("Testing matrix size: " + size + "x" + size);
                System.out.println("------------------------------------------------------------");
                
                BenchmarkResult result = null;
                try {
                    result = benchmarkSize(size, multiplier, distributedAvailable);
                    allResults.add(result);
                    printSizeSummary(result);
                } catch (InterruptedException e) {
                    System.err.println("Benchmark interrupted: " + e.getMessage());
                    Thread.currentThread().interrupt();
                }
                System.out.println();
            }
            
            // Print overall summary
            printOverallSummary(allResults);
            
            // Save results to file with method suffix
            String methodSuffix = METHOD.equals("all") ? null : METHOD;
            int clusterSize = distributedAvailable ? multiplier.getClusterSize() : 0;
            saveResults(allResults, methodSuffix, clusterSize);
            
        } finally {
            if (distributedAvailable) {
                multiplier.disconnect();
            }
        }
    }
    
    private static BenchmarkResult benchmarkSize(int size, DistributedMatrixMultiplier multiplier, boolean distributedAvailable) throws InterruptedException {
        BenchmarkResult result = new BenchmarkResult(size);
        Runtime runtime = Runtime.getRuntime();
        
        // Generate test matrices
        double[][] A = DistributedMatrixMultiplier.generateRandomMatrix(size, size);
        double[][] B = DistributedMatrixMultiplier.generateRandomMatrix(size, size);
        
        if (METHOD.equals("all") || METHOD.equals("basic")) {
            // Basic multiplication
            System.out.println("  Testing basic multiplication (" + NUM_RUNS + " runs)...");
            // Warmup runs (3 iterations for JIT compilation)
            System.out.print("  Warmup: ");
            for (int w = 0; w < 3; w++) {
                DistributedMatrixMultiplier.basicMultiply(A, B);
                System.gc();
                System.out.print("." + (w + 1));
            }
            System.out.println(" done");
            Thread.sleep(100);
            
            result.basicTimes = new double[NUM_RUNS];
            result.memoryUsed = new double[NUM_RUNS];
            for (int i = 0; i < NUM_RUNS; i++) {
                System.gc();
                System.gc();
                Thread.sleep(100);
                long memBefore = runtime.totalMemory() - runtime.freeMemory();
                long start = System.nanoTime();
                double[][] resultMatrix = DistributedMatrixMultiplier.basicMultiply(A, B);
                long memAfter = runtime.totalMemory() - runtime.freeMemory();
                long end = System.nanoTime();
                result.basicTimes[i] = (end - start) / 1_000_000.0;
                result.memoryUsed[i] = Math.max(0.0, (memAfter - memBefore) / 1024.0);
                // Keep reference to prevent GC
                if (resultMatrix[0][0] < -999999) System.out.println("");
            }
        } else {
            result.basicTimes = new double[0];
            result.memoryUsed = new double[0];
        }
        
        if (METHOD.equals("all") || METHOD.equals("parallel")) {
            // Parallel multiplication
            System.out.println("  Testing parallel multiplication (" + NUM_RUNS + " runs)...");
            // Warmup runs (3 iterations for JIT compilation)
            System.out.print("  Warmup: ");
            for (int w = 0; w < 3; w++) {
                DistributedMatrixMultiplier.parallelMultiply(A, B);
                System.gc();
                System.out.print("." + (w + 1));
            }
            System.out.println(" done");
            Thread.sleep(100);
            
            result.parallelTimes = new double[NUM_RUNS];
            result.parallelMemory = new double[NUM_RUNS];
            for (int i = 0; i < NUM_RUNS; i++) {
                System.gc();
                System.gc();
                Thread.sleep(100);
                long memBefore = runtime.totalMemory() - runtime.freeMemory();
                long start = System.nanoTime();
                double[][] resultMatrix = DistributedMatrixMultiplier.parallelMultiply(A, B);
                long memAfter = runtime.totalMemory() - runtime.freeMemory();
                long end = System.nanoTime();
                result.parallelTimes[i] = (end - start) / 1_000_000.0;
                result.parallelMemory[i] = Math.max(0.0, (memAfter - memBefore) / 1024.0);
                // Keep reference to prevent GC
                if (resultMatrix[0][0] < -999999) System.out.println("");
            }
        } else {
            result.parallelTimes = new double[0];
            result.parallelMemory = new double[0];
        }
        
        // Distributed multiplication
        if (distributedAvailable) {
            System.out.println("  Testing distributed multiplication (" + NUM_RUNS + " runs)...");
            // Warmup runs (3 iterations for JIT compilation)
            System.out.print("  Warmup: ");
            for (int w = 0; w < 3; w++) {
                multiplier.multiply(A, B);
                System.gc();
                System.out.print("." + (w + 1));
            }
            System.out.println(" done");
            Thread.sleep(100);
            
            result.distributedTimes = new double[NUM_RUNS];
            result.networkTimes = new double[NUM_RUNS];
            result.dataTransferred = new double[NUM_RUNS];
            result.distributedMemory = new double[NUM_RUNS];
            result.clusterNodes = multiplier.getClusterSize();
            result.memoryPerNodeKB = 0.0;
            
            for (int i = 0; i < NUM_RUNS; i++) {
                System.gc();
                System.gc();
                Thread.sleep(100);
                long memBefore = runtime.totalMemory() - runtime.freeMemory();
                long start = System.nanoTime();
                double[][] resultMatrix = multiplier.multiply(A, B);
                long memAfter = runtime.totalMemory() - runtime.freeMemory();
                long end = System.nanoTime();
                result.distributedTimes[i] = (end - start) / 1_000_000.0;
                result.networkTimes[i] = multiplier.getNetworkTimeMs();
                result.dataTransferred[i] = multiplier.getDataTransferredMB();
                result.distributedMemory[i] = Math.max(0.0, (memAfter - memBefore) / 1024.0);
                // Keep reference to prevent GC
                if (resultMatrix[0][0] < -999999) System.out.println("");
            }
            
            result.clusterNodes = multiplier.getClusterSize();
            result.memoryPerNodeKB = multiplier.getMemoryPerNodeMB() * 1024; // Convert MB to KB
        }
        
        return result;
    }
    
    private static long[] measurePerformance(Runnable task, int numRuns) {
        long[] times = new long[numRuns];
        
        for (int i = 0; i < numRuns; i++) {
            long start = System.currentTimeMillis();
            task.run();
            long end = System.currentTimeMillis();
            times[i] = end - start;
        }
        
        return times;
    }
    
    private static void printSizeSummary(BenchmarkResult result) {
        System.out.println("\n  Results for " + result.size + "x" + result.size + " matrices:");
        System.out.println("    Basic:       " + formatTime(result.basicTimes) + 
                          " | Memory: " + String.format("%.2f KB", averageDouble(result.memoryUsed)));
        System.out.println("    Parallel:    " + formatTime(result.parallelTimes) + 
                          " | Memory: " + String.format("%.2f KB", averageDouble(result.parallelMemory)));
        
        if (result.distributedTimes != null) {
            double avgNetworkTime = averageDouble(result.networkTimes);
            double avgExecTime = averageDouble(result.distributedTimes);
            double networkPercent = (avgNetworkTime / avgExecTime) * 100;
            double avgDataTransferred = averageDouble(result.dataTransferred);
            double avgMemory = averageDouble(result.distributedMemory);
            
            System.out.println("    Distributed: " + formatTime(result.distributedTimes));
            System.out.println("                 Network: " + String.format("%.2fms (%.1f%% overhead) | Data: %.2f MB transferred", 
                              avgNetworkTime, networkPercent, avgDataTransferred));
            System.out.println("                 Cluster: " + result.clusterNodes + " node(s) | Memory: " + String.format("%.2f KB", avgMemory));
        } else {
            System.out.println("    Distributed: Not available");
        }
    }
    
    private static String formatTime(double[] times) {
        if (times == null || times.length == 0) {
            return "N/A";
        }
        
        double avg = averageDouble(times);
        double min = minDouble(times);
        double max = maxDouble(times);
        
        return String.format("%.2fms (min: %.2fms, max: %.2fms)", 
                           avg, min, max);
    }
    
    private static void printOverallSummary(List<BenchmarkResult> results) {
        System.out.println("============================================================");
        System.out.println("OVERALL SUMMARY");
        System.out.println("============================================================");
        System.out.println();
        
        System.out.printf("%-12s %-15s %-15s %-15s%n", "Matrix Size", "Basic (ms)", "Parallel (ms)", "Distributed (ms)");
        System.out.println("------------------------------------------------------------");
        
        for (BenchmarkResult result : results) {
            System.out.printf("%-12s %-15s %-15s %-15s%n",
                result.size + "x" + result.size,
                String.format("%.2fms", averageDouble(result.basicTimes)),
                String.format("%.2fms", averageDouble(result.parallelTimes)),
                result.distributedTimes != null ? 
                    String.format("%.2fms", averageDouble(result.distributedTimes)) : "N/A"
            );
        }
        
        System.out.println("============================================================");
    }
    
    private static void saveResults(List<BenchmarkResult> results, String methodSuffix, int clusterSize) {
        // Create results directory if it doesn't exist (relative to working directory)
        String resultsDir = System.getProperty("user.dir") + "/results";
        new java.io.File(resultsDir).mkdirs();
        
        String filename = resultsDir + "/benchmark_results_java";
        if (methodSuffix != null && !methodSuffix.isEmpty()) {
            filename += "_" + methodSuffix;
        }
        if (clusterSize > 0) {
            filename += "_" + clusterSize + "nodes";
        }
        filename += "_" + System.currentTimeMillis() + ".csv";
        
        try (FileWriter writer = new FileWriter(filename)) {
            // Write header
            writer.write("matrix_size,method,run,time_ms,network_time_ms,data_transferred_kb,memory_kb,cluster_nodes,memory_per_node_kb\n");
            
            // Write data
            for (BenchmarkResult result : results) {
                // Basic times
                for (int i = 0; i < result.basicTimes.length; i++) {
                    writer.write(result.size + ",basic," + (i+1) + "," + String.format(Locale.US, "%.2f", result.basicTimes[i]) + ",0,0," + 
                                String.format(Locale.US, "%.2f", result.memoryUsed[i]) + ",0,0\n");
                }
                
                // Parallel times
                for (int i = 0; i < result.parallelTimes.length; i++) {
                    writer.write(result.size + ",parallel," + (i+1) + "," + String.format(Locale.US, "%.2f", result.parallelTimes[i]) + ",0,0," + 
                                String.format(Locale.US, "%.2f", result.parallelMemory[i]) + ",0,0\n");
                }
                
                // Distributed times
                if (result.distributedTimes != null) {
                    for (int i = 0; i < result.distributedTimes.length; i++) {
                        writer.write(result.size + ",distributed," + (i+1) + "," + String.format(Locale.US, "%.2f", result.distributedTimes[i]) + "," +
                                    String.format(Locale.US, "%.2f", result.networkTimes[i]) + "," + String.format(Locale.US, "%.2f", result.dataTransferred[i] * 1024) + "," + 
                                    String.format(Locale.US, "%.2f", result.distributedMemory[i]) + "," + result.clusterNodes + "," +
                                    String.format(Locale.US, "%.2f", result.memoryPerNodeKB) + "\n");
                    }
                }
            }
            
            System.out.println("\nResults saved to: " + filename);
            
        } catch (IOException e) {
            System.err.println("Error saving results: " + e.getMessage());
        }
    }
    
    // Helper methods
    private static double average(long[] values) {
        long sum = 0;
        for (long value : values) {
            sum += value;
        }
        return (double) sum / values.length;
    }
    
    private static long min(long[] values) {
        long min = values[0];
        for (long value : values) {
            if (value < min) min = value;
        }
        return min;
    }
    
    private static long max(long[] values) {
        long max = values[0];
        for (long value : values) {
            if (value > max) max = value;
        }
        return max;
    }
    
    private static String arrayToString(int[] arr) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < arr.length; i++) {
            sb.append(arr[i]);
            if (i < arr.length - 1) sb.append(", ");
        }
        return sb.toString();
    }
    
    private static double averageDouble(double[] values) {
        double sum = 0;
        for (double value : values) {
            sum += value;
        }
        return sum / values.length;
    }
    
    private static double minDouble(double[] values) {
        double min = values[0];
        for (double value : values) {
            if (value < min) min = value;
        }
        return min;
    }
    
    private static double maxDouble(double[] values) {
        double max = values[0];
        for (double value : values) {
            if (value > max) max = value;
        }
        return max;
    }
    
    private static double[] convertToDouble(long[] values) {
        double[] result = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            result[i] = (double) values[i];
        }
        return result;
    }
    
    // Inner class to store results
    private static class BenchmarkResult {
        int size;
        double[] basicTimes;
        double[] parallelTimes;
        double[] distributedTimes;
        double[] networkTimes;
        double[] dataTransferred;
        double[] memoryUsed;          // For basic (KB with decimals)
        double[] parallelMemory;      // For parallel (KB with decimals)
        double[] distributedMemory;   // For distributed (KB with decimals)
        int clusterNodes;
        double memoryPerNodeKB;     // Estimated memory per cluster node in KB
        
        BenchmarkResult(int size) {
            this.size = size;
        }
    }
}