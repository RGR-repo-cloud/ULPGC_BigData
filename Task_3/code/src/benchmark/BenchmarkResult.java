package benchmark;

/**
 * Data structure to store benchmark results for analysis.
 */
public class BenchmarkResult {
    
    private final String algorithmName;
    private final int matrixSize;
    private final long executionTimeNanos;
    private final int numThreads;
    private final long memoryUsedBytes;
    
    public BenchmarkResult(String algorithmName, int matrixSize, 
                          long executionTimeNanos, int numThreads, 
                          long memoryUsedBytes) {
        this.algorithmName = algorithmName;
        this.matrixSize = matrixSize;
        this.executionTimeNanos = executionTimeNanos;
        this.numThreads = numThreads;
        this.memoryUsedBytes = memoryUsedBytes;
    }
    
    public String getAlgorithmName() {
        return algorithmName;
    }
    
    public int getMatrixSize() {
        return matrixSize;
    }
    
    public long getExecutionTimeNanos() {
        return executionTimeNanos;
    }
    
    public double getExecutionTimeMillis() {
        return executionTimeNanos / 1_000_000.0;
    }
    
    public double getExecutionTimeSeconds() {
        return executionTimeNanos / 1_000_000_000.0;
    }
    
    public int getNumThreads() {
        return numThreads;
    }
    
    public long getMemoryUsedBytes() {
        return memoryUsedBytes;
    }
    
    public double getMemoryUsedMB() {
        return memoryUsedBytes / (1024.0 * 1024.0);
    }
    

    
    /**
     * Calculates speedup compared to a baseline result.
     * 
     * @param baseline The baseline result to compare against
     * @return Speedup factor (baseline_time / this_time)
     */
    public double getSpeedup(BenchmarkResult baseline) {
        return (double) baseline.getExecutionTimeNanos() / this.executionTimeNanos;
    }
    
    /**
     * Calculates parallel efficiency (speedup per thread).
     * 
     * @param baseline The sequential baseline result
     * @return Efficiency (speedup / number_of_threads)
     */
    public double getEfficiency(BenchmarkResult baseline) {
        if (numThreads <= 1) return 1.0;
        return getSpeedup(baseline) / numThreads;
    }
    
    @Override
    public String toString() {
        return String.format("Algorithm: %s, Size: %dx%d, Time: %.2f ms, Threads: %d, Memory: %.2f MB",
                algorithmName, matrixSize, matrixSize, getExecutionTimeMillis(), 
                numThreads, getMemoryUsedMB());
    }
}