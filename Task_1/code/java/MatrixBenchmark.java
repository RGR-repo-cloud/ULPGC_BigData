/**
 * Java Matrix Multiplication Benchmark
 */
public class MatrixBenchmark {
    
    /**
     * Class to hold benchmark results
     */
    public static class BenchmarkResult {
        public final double[] times;
        public final double[] memoryUsage;
        
        public BenchmarkResult(double[] times, double[] memoryUsage) {
            this.times = times;
            this.memoryUsage = memoryUsage;
        }
    }
    
    /**
     * Benchmark matrix multiplication for a given size
     */
    public static BenchmarkResult benchmarkMatrixMultiply(int size, int numRuns) {
        double[] times = new double[numRuns];
        double[] memoryUsage = new double[numRuns];
        Runtime runtime = Runtime.getRuntime();
        
        for (int run = 0; run < numRuns; run++) {
            // Force garbage collection before test
            System.gc();
            
            // Measure memory before
            long memoryBefore = runtime.totalMemory() - runtime.freeMemory();
            
            // Create matrices for this run using utilities
            double[][] A = MatrixUtils.createRandomMatrix(size);
            double[][] B = MatrixUtils.createRandomMatrix(size);
            
            // Measure execution time
            long startTime = System.nanoTime();
            MatrixMultiplier.matrixMultiply(A, B);
            long endTime = System.nanoTime();
            
            // Measure memory after
            long memoryAfter = runtime.totalMemory() - runtime.freeMemory();
            double memoryUsedMB = (memoryAfter - memoryBefore) / 1024.0 / 1024.0;
            
            double executionTime = (endTime - startTime) / 1e9; // Convert to seconds
            times[run] = executionTime;
            memoryUsage[run] = Math.max(0, memoryUsedMB); // Ensure non-negative
        }
        
        return new BenchmarkResult(times, memoryUsage);
    }
    
    /**
     * Run benchmark tests for different matrix sizes
     */
    public static void runBenchmark() {
        // Test different matrix sizes
        int[] sizes = {64, 128, 256, 512, 1024};
        int numRuns = 5;
        
        // Check for CSV file argument
        String csvFile = null;
        String[] args = System.getProperty("csv.file", "").split(",");
        if (args.length > 0 && !args[0].isEmpty()) {
            csvFile = args[0];
        }
        
        System.out.println("Java Matrix Multiplication Benchmark");
        System.out.println("==================================================");
        System.out.println("Number of runs per size: " + numRuns);
        System.out.print("Matrix sizes: ");
        for (int i = 0; i < sizes.length; i++) {
            System.out.print(sizes[i]);
            if (i < sizes.length - 1) System.out.print(", ");
        }
        System.out.println();
        if (csvFile != null) {
            System.out.println("CSV output: " + csvFile);
        }
        System.out.println();
        
        // Store all results for CSV output
        double[][] allTimes = new double[sizes.length][];
        double[][] allMemory = new double[sizes.length][];
        
        
        for (int s = 0; s < sizes.length; s++) {
            int size = sizes[s];
            System.out.println("Testing matrix size " + size + "x" + size + "...");
            
            BenchmarkResult result = benchmarkMatrixMultiply(size, numRuns);
            double[] times = result.times;
            double[] memoryUsage = result.memoryUsage;
            
            // Store for CSV output
            allTimes[s] = times;
            allMemory[s] = memoryUsage;
            
            // Calculate statistics using utility functions
            double avgTime = StatsUtils.calculateAverage(times);
            double minTime = StatsUtils.findMin(times);
            double maxTime = StatsUtils.findMax(times);
            double stdDev = StatsUtils.calculateStdDev(times);
            
            double avgMemory = StatsUtils.calculateAverage(memoryUsage);
            double maxMemory = StatsUtils.findMax(memoryUsage);
            
            System.out.print("  Times: [");
            for (int i = 0; i < times.length; i++) {
                System.out.printf("%.4f", times[i]);
                if (i < times.length - 1) System.out.print(", ");
            }
            System.out.println("] seconds");
            
            System.out.printf("  Average: %.4f seconds%n", avgTime);
            System.out.printf("  Min: %.4f seconds%n", minTime);
            System.out.printf("  Max: %.4f seconds%n", maxTime);
            System.out.printf("  Std Dev: %.4f seconds%n", stdDev);
            
            System.out.print("  Memory: [");
            for (int i = 0; i < memoryUsage.length; i++) {
                System.out.printf("%.1f", memoryUsage[i]);
                if (i < memoryUsage.length - 1) System.out.print(", ");
            }
            System.out.println("] MB");
            
            System.out.printf("  Avg Memory: %.1f MB%n", avgMemory);
            System.out.printf("  Peak Memory: %.1f MB%n", maxMemory);
            System.out.println();
        }
        
        // Write to CSV if specified
        if (csvFile != null) {
            CSVWriter.writeBenchmarkToCSV("Java", sizes, allTimes, allMemory, csvFile);
        }
    }
    
    public static void main(String[] args) {
        // Set CSV file if provided as argument
        if (args.length > 0) {
            System.setProperty("csv.file", args[0]);
        }
        runBenchmark();
    }
}