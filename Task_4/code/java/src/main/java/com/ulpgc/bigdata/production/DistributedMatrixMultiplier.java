package com.ulpgc.bigdata.production;

import com.hazelcast.client.HazelcastClient;
import com.hazelcast.client.config.ClientConfig;
import com.hazelcast.core.HazelcastInstance;
import com.hazelcast.map.IMap;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.logging.Logger;

/**
 * Distributed matrix multiplication using Hazelcast In-Memory Data Grid.
 * Implements block-wise matrix multiplication for scalability.
 */
public class DistributedMatrixMultiplier {
    private static final Logger logger = Logger.getLogger(DistributedMatrixMultiplier.class.getName());
    
    private HazelcastInstance client;
    private final String clusterName;
    private final int blockSize;
    
    // Metrics tracking
    private double networkTimeMs = 0.0;
    private double dataTransferredMB = 0.0;
    private int clusterSize = 0;
    private double memoryPerNodeMB = 0.0;
    
    public DistributedMatrixMultiplier(String clusterName) {
        this.clusterName = clusterName;
        this.blockSize = 64; // Default block size
    }
    
    public DistributedMatrixMultiplier() {
        this("dev");
    }
    
    /**
     * Connect to Hazelcast cluster.
     */
    public boolean connect() {
        try {
            ClientConfig config = new ClientConfig();
            config.setClusterName(clusterName);
            config.getNetworkConfig().addAddress("127.0.0.1:5701", "127.0.0.1:5702", "127.0.0.1:5703");
            client = HazelcastClient.newHazelcastClient(config);
            clusterSize = client.getCluster().getMembers().size();
            logger.info("Connected to Hazelcast cluster with " + clusterSize + " node(s)");
            return true;
        } catch (Exception e) {
            logger.severe("Failed to connect: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * Disconnect from Hazelcast cluster.
     */
    public void disconnect() {
        if (client != null) {
            client.shutdown();
            logger.info("Disconnected from cluster");
        }
    }
    
    /**
     * Store matrix as distributed blocks.
     */
    private List<BlockIndex> storeMatrixBlocks(double[][] matrix, String mapName) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        List<BlockIndex> blockIndices = new ArrayList<>();
        
        IMap<String, double[][]> matrixMap = client.getMap(mapName);
        
        // Split matrix into blocks
        for (int i = 0; i < rows; i += blockSize) {
            for (int j = 0; j < cols; j += blockSize) {
                int endI = Math.min(i + blockSize, rows);
                int endJ = Math.min(j + blockSize, cols);
                
                // Extract block
                double[][] block = new double[endI - i][endJ - j];
                for (int bi = 0; bi < endI - i; bi++) {
                    for (int bj = 0; bj < endJ - j; bj++) {
                        block[bi][bj] = matrix[i + bi][j + bj];
                    }
                }
                
                String key = i + "_" + j;
                matrixMap.put(key, block);
                blockIndices.add(new BlockIndex(i, j));
            }
        }
        
        return blockIndices;
    }
    
    /**
     * Retrieve matrix block from distributed storage.
     */
    private double[][] getMatrixBlock(String mapName, int i, int j) {
        IMap<String, double[][]> matrixMap = client.getMap(mapName);
        String key = i + "_" + j;
        double[][] block = matrixMap.get(key);
        return block != null ? block : new double[blockSize][blockSize];
    }
    
    /**
     * Perform distributed matrix multiplication.
     */
    public double[][] multiply(double[][] A, double[][] B) {
        if (A[0].length != B.length) {
            throw new IllegalArgumentException("Matrix dimensions don't match for multiplication");
        }
        
        // Reset metrics
        networkTimeMs = 0.0;
        dataTransferredMB = 0.0;
        
        int m = A.length;
        int n = A[0].length;
        int p = B[0].length;
        
        logger.info(String.format("Multiplying matrices: (%dx%d) * (%dx%d)", m, n, n, p));
        
        // Store matrices in distributed storage
        logger.info("Storing matrices in distributed storage...");
        long startTime = System.nanoTime();
        
        storeMatrixBlocks(A, "matrix_A");
        storeMatrixBlocks(B, "matrix_B");
        
        long storageTime = System.nanoTime() - startTime;
        networkTimeMs += storageTime / 1_000_000.0;
        
        // Calculate data size for matrices A and B
        double dataSizeMB = ((double)(m * n + n * p) * 8) / (1024 * 1024);
        dataTransferredMB += dataSizeMB;
        
        // Estimate memory per node (data is distributed across cluster)
        memoryPerNodeMB = dataSizeMB / Math.max(clusterSize, 1);
        
        logger.info(String.format("Storage time: %.2fs, Data uploaded: %.2f MB", 
                                  storageTime / 1000.0, dataSizeMB));
        logger.info(String.format("Estimated memory per node: %.2f MB", memoryPerNodeMB));
        
        // Initialize result matrix
        double[][] C = new double[m][p];
        
        // Perform block-wise multiplication
        logger.info("Performing distributed multiplication...");
        long computeStart = System.nanoTime();
        
        int blocksRetrieved = 0;
        for (int i = 0; i < m; i += blockSize) {
            for (int j = 0; j < p; j += blockSize) {
                // Compute C[i:i+blockSize, j:j+blockSize]
                int resultRows = Math.min(blockSize, m - i);
                int resultCols = Math.min(blockSize, p - j);
                double[][] resultBlock = new double[resultRows][resultCols];
                
                for (int k = 0; k < n; k += blockSize) {
                    // Get blocks A[i:i+bs, k:k+bs] and B[k:k+bs, j:j+bs]
                    // Time only the network operations (block retrieval)
                    long networkOpStart = System.nanoTime();
                    double[][] blockA = getMatrixBlock("matrix_A", i, k);
                    double[][] blockB = getMatrixBlock("matrix_B", k, j);
                    long networkOpTime = System.nanoTime() - networkOpStart;
                    networkTimeMs += networkOpTime / 1_000_000.0;
                    
                    blocksRetrieved += 2;
                    
                    // Multiply blocks and add to result (computation, not network)
                    multiplyAndAdd(blockA, blockB, resultBlock, 
                                 Math.min(blockSize, m - i),
                                 Math.min(blockSize, n - k),
                                 Math.min(blockSize, p - j));
                }
                
                // Store result in final matrix
                for (int bi = 0; bi < resultRows; bi++) {
                    for (int bj = 0; bj < resultCols; bj++) {
                        C[i + bi][j + bj] = resultBlock[bi][bj];
                    }
                }
            }
        }
        
        // Estimate data transfer for block retrievals (conservative estimate)
        double blockDataMB = (blocksRetrieved * blockSize * blockSize * 8.0) / (1024 * 1024);
        dataTransferredMB += blockDataMB;
        
        long computeTime = System.nanoTime() - computeStart;
        logger.info(String.format("Computation time: %.2fs", computeTime / 1_000_000_000.0));
        logger.info(String.format("Blocks retrieved: %d", blocksRetrieved));
        
        // Cleanup distributed storage
        long cleanupStart = System.nanoTime();
        client.getMap("matrix_A").clear();
        client.getMap("matrix_B").clear();
        long cleanupTime = System.nanoTime() - cleanupStart;
        networkTimeMs += cleanupTime / 1_000_000.0;
        
        logger.info(String.format("Total network time: %.2fs, Total data transferred: %.2f MB", 
                                  networkTimeMs / 1000.0, dataTransferredMB));
        
        return C;
    }
    
    /**
     * Helper method to multiply two blocks and add to result.
     */
    private void multiplyAndAdd(double[][] A, double[][] B, double[][] result, 
                               int rows, int inner, int cols) {
        for (int i = 0; i < rows && i < A.length; i++) {
            for (int j = 0; j < cols && j < B[0].length; j++) {
                for (int k = 0; k < inner && k < A[0].length && k < B.length; k++) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    }
    
    /**
     * Get network time in milliseconds.
     */
    public double getNetworkTimeMs() {
        return networkTimeMs;
    }
    
    /**
     * Get total data transferred in MB.
     */
    public double getDataTransferredMB() {
        return dataTransferredMB;
    }
    
    /**
     * Get number of nodes in cluster.
     */
    public int getClusterSize() {
        return clusterSize;
    }
    
    /**
     * Get estimated memory per node in MB.
     */
    public double getMemoryPerNodeMB() {
        return memoryPerNodeMB;
    }
    
    /**
     * Basic sequential matrix multiplication for comparison.
     */
    public static double[][] basicMultiply(double[][] A, double[][] B) {
        int m = A.length;
        int n = A[0].length;
        int p = B[0].length;
        
        double[][] C = new double[m][p];
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                for (int k = 0; k < n; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        
        return C;
    }
    
    /**
     * Parallel matrix multiplication using basic threading.
     */
    public static double[][] parallelMultiply(double[][] A, double[][] B) {
        int m = A.length;
        int n = A[0].length;
        int p = B[0].length;
        
        double[][] C = new double[m][p];
        
        // Determine number of threads
        int numThreads = Runtime.getRuntime().availableProcessors();
        int rowsPerThread = m / numThreads;
        
        Thread[] threads = new Thread[numThreads];
        
        // Create and start threads
        for (int t = 0; t < numThreads; t++) {
            final int startRow = t * rowsPerThread;
            final int endRow = (t == numThreads - 1) ? m : (t + 1) * rowsPerThread;
            
            threads[t] = new Thread(() -> {
                // Compute matrix multiplication for this range of rows
                for (int i = startRow; i < endRow; i++) {
                    for (int j = 0; j < p; j++) {
                        for (int k = 0; k < n; k++) {
                            C[i][j] += A[i][k] * B[k][j];
                        }
                    }
                }
            });
            
            threads[t].start();
        }
        
        // Wait for all threads to complete
        for (Thread thread : threads) {
            try {
                thread.join();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("Thread interrupted during parallel multiplication", e);
            }
        }
        
        return C;
    }
    
    /**
     * Generate random matrix for testing.
     */
    public static double[][] generateRandomMatrix(int rows, int cols) {
        double[][] matrix = new double[rows][cols];
        Random random = new Random();
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = random.nextDouble();
            }
        }
        
        return matrix;
    }
    
    /**
     * Check if two matrices are approximately equal.
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
    
    public static void main(String[] args) {
        DistributedMatrixMultiplier multiplier = new DistributedMatrixMultiplier();
        
        if (!multiplier.connect()) {
            System.out.println("Failed to connect to Hazelcast. Make sure Hazelcast server is running.");
            System.exit(1);
        }
        
        try {
            // Test with small matrices
            double[][] A = generateRandomMatrix(100, 80);
            double[][] B = generateRandomMatrix(80, 60);
            
            System.out.println("Testing distributed matrix multiplication...");
            long startTime = System.currentTimeMillis();
            double[][] result = multiplier.multiply(A, B);
            long distTime = System.currentTimeMillis() - startTime;
            
            System.out.printf("Distributed multiplication completed in %.2fs%n", distTime / 1000.0);
            System.out.printf("Result shape: %dx%d%n", result.length, result[0].length);
            
            // Verify correctness
            System.out.println("\nVerifying with basic multiplication...");
            double[][] expected = basicMultiply(A, B);
            if (matricesEqual(result, expected, 1e-10)) {
                System.out.println("✓ Distributed result is correct!");
            } else {
                System.out.println("✗ Distributed result verification failed!");
            }
            
            // Test parallel multiplication
            System.out.println("\nTesting parallel multiplication...");
            startTime = System.currentTimeMillis();
            double[][] parallelResult = parallelMultiply(A, B);
            long parallelTime = System.currentTimeMillis() - startTime;
            
            System.out.printf("Parallel multiplication completed in %.2fs%n", parallelTime / 1000.0);
            
            if (matricesEqual(parallelResult, expected, 1e-10)) {
                System.out.println("✓ Parallel result is correct!");
            } else {
                System.out.println("✗ Parallel result verification failed!");
            }
            
        } finally {
            multiplier.disconnect();
        }
    }
    
    /**
     * Helper class to store block indices.
     */
    private static class BlockIndex {
        final int i, j;
        
        BlockIndex(int i, int j) {
            this.i = i;
            this.j = j;
        }
    }
}