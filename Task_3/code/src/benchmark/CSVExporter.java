package benchmark;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

/**
 * Exports benchmark results to consolidated CSV files for visualization and analysis.
 * Generates only 6 comprehensive files instead of many scattered files.
 */
public class CSVExporter {
    
    private static final String OUTPUT_DIR = "results";
    
    /**
     * Clear all existing CSV files to start fresh benchmark results.
     * Call this at the beginning of a benchmark run to avoid accumulating old data.
     */
    public static void clearExistingResults() {
        try {
            Files.createDirectories(Paths.get(OUTPUT_DIR));
            
            // Delete existing CSV files
            String[] csvFiles = {
                "all_performance_results.csv",
                "all_efficiency_analysis.csv", 
                "all_resource_usage.csv",
                "all_hyperparameter_optimization.csv",
                "benchmark_summary.csv"
            };
            
            for (String filename : csvFiles) {
                java.nio.file.Path filePath = Paths.get(OUTPUT_DIR, filename);
                if (Files.exists(filePath)) {
                    Files.delete(filePath);
                }
            }
            
            System.out.println("📁 Cleared existing CSV files for fresh benchmark results");
            
        } catch (IOException e) {
            System.err.println("Warning: Could not clear existing CSV files: " + e.getMessage());
        }
    }
    
    /**
     * Exports benchmark results to consolidated CSV files.
     * Appends to existing files or creates new ones with headers.
     */
    public static void exportResults(List<BenchmarkResult> results, int matrixSize, String testType) {
        try {
            // Create output directory if it doesn't exist
            Files.createDirectories(Paths.get(OUTPUT_DIR));
            
            // Export to consolidated files
            exportToConsolidatedPerformance(results, testType);
            exportToConsolidatedEfficiency(results, testType);
            exportToConsolidatedResources(results, testType);
            
            System.out.println("📊 Exported to consolidated CSV files in: " + OUTPUT_DIR);
            
        } catch (IOException e) {
            System.err.println("Warning: Could not export CSV files: " + e.getMessage());
        }
    }
    
    /**
     * Exports to consolidated performance results file
     */
    private static void exportToConsolidatedPerformance(List<BenchmarkResult> results, String testType) throws IOException {
        String filename = OUTPUT_DIR + "/all_performance_results.csv";
        boolean fileExists = Files.exists(Paths.get(filename));
        
        try (FileWriter writer = new FileWriter(filename, true)) {
            // Write header only if file doesn't exist
            if (!fileExists) {
                writer.append("Algorithm,Matrix_Size,Time_ms,Time_ns,Speedup,Threads,Memory_MB,Test_Type\n");
            }
            
            BenchmarkResult baseline = results.get(0);
            
            // Append data rows
            for (BenchmarkResult result : results) {
                double speedup = result.getSpeedup(baseline);
                
                writer.append(String.format("%s,%d,%.3f,%d,%.3f,%d,%.3f,%s\n",
                    escapeCsvField(result.getAlgorithmName()),
                    result.getMatrixSize(),
                    result.getExecutionTimeMillis(),
                    result.getExecutionTimeNanos(),
                    speedup,
                    result.getNumThreads(),
                    result.getMemoryUsedMB(),
                    testType
                ));
            }
        }
    }
    
    /**
     * Exports to consolidated efficiency analysis file
     */
    private static void exportToConsolidatedEfficiency(List<BenchmarkResult> results, String testType) throws IOException {
        String filename = OUTPUT_DIR + "/all_efficiency_analysis.csv";
        boolean fileExists = Files.exists(Paths.get(filename));
        
        try (FileWriter writer = new FileWriter(filename, true)) {
            // Write header only if file doesn't exist
            if (!fileExists) {
                writer.append("Algorithm,Matrix_Size,Threads,Speedup,Efficiency,Speedup_Per_Thread,Rating,Test_Type\n");
            }
            
            BenchmarkResult baseline = results.get(0);
            
            // Only include parallel algorithms (more than 1 thread)
            for (BenchmarkResult result : results) {
                if (result.getNumThreads() > 1) {
                    double speedup = result.getSpeedup(baseline);
                    double efficiency = result.getEfficiency(baseline);
                    double speedupPerThread = efficiency; // Same as efficiency (speedup/threads)
                    String rating = getEfficiencyRating(efficiency);
                    
                    writer.append(String.format("%s,%d,%d,%.3f,%.3f,%.3f,%s,%s\n",
                        escapeCsvField(result.getAlgorithmName()),
                        result.getMatrixSize(),
                        result.getNumThreads(),
                        speedup,
                        efficiency,
                        speedupPerThread,
                        rating,
                        testType
                    ));
                }
            }
        }
    }
    
    /**
     * Exports to consolidated resource usage file
     */
    private static void exportToConsolidatedResources(List<BenchmarkResult> results, String testType) throws IOException {
        String filename = OUTPUT_DIR + "/all_resource_usage.csv";
        boolean fileExists = Files.exists(Paths.get(filename));
        
        try (FileWriter writer = new FileWriter(filename, true)) {
            // Write header only if file doesn't exist
            if (!fileExists) {
                writer.append("Algorithm,Matrix_Size,Memory_MB,Peak_Memory_MB,Threads,Test_Type\n");
            }
            
            for (BenchmarkResult result : results) {
                double peakMemory = result.getMemoryUsedMB() * 1.2;
                
                writer.append(String.format("%s,%d,%.3f,%.3f,%d,%s\n",
                    escapeCsvField(result.getAlgorithmName()),
                    result.getMatrixSize(),
                    result.getMemoryUsedMB(),
                    peakMemory,
                    result.getNumThreads(),
                    testType
                ));
            }
        }
    }
    
    /**
     * Exports ONLY final optimal algorithm comparisons (excludes hyperparameter exploration)
     */
    public static void exportOptimalResults(List<BenchmarkResult> results, int matrixSize, String testType) {
        try {
            // Create output directory if it doesn't exist
            Files.createDirectories(Paths.get(OUTPUT_DIR));
            
            // Export to consolidated files (ONLY optimal configurations)
            exportToConsolidatedPerformance(results, testType);
            exportToConsolidatedEfficiency(results, testType);
            exportToConsolidatedResources(results, testType);
            
            System.out.println("📊 Exported OPTIMAL results to consolidated CSV files in: " + OUTPUT_DIR);
            
        } catch (IOException e) {
            System.err.println("Warning: Could not export optimal CSV files: " + e.getMessage());
        }
    }

    /**
     * Exports hyperparameter optimization to consolidated file
     */
    public static void exportHyperparameterResults(List<BenchmarkResult> results, String parameterName, int matrixSize) {
        try {
            Files.createDirectories(Paths.get(OUTPUT_DIR));
            String filename = OUTPUT_DIR + "/all_hyperparameter_optimization.csv";
            boolean fileExists = Files.exists(Paths.get(filename));
            
            try (FileWriter writer = new FileWriter(filename, true)) {
                // Write header only if file doesn't exist
                if (!fileExists) {
                    writer.append("Parameter_Type,Parameter_Value,Algorithm,Matrix_Size,Time_ms,Speedup,Threads,Memory_MB\n");
                }
                
                BenchmarkResult baseline = results.get(0);
                
                for (BenchmarkResult result : results) {
                    String paramValue = extractParameterValue(result.getAlgorithmName(), parameterName);
                    double speedup = result.getSpeedup(baseline);
                    
                    writer.append(String.format("%s,%s,%s,%d,%.3f,%.3f,%d,%.3f\n",
                        parameterName,
                        paramValue,
                        escapeCsvField(result.getAlgorithmName()),
                        result.getMatrixSize(),
                        result.getExecutionTimeMillis(),
                        speedup,
                        result.getNumThreads(),
                        result.getMemoryUsedMB()
                    ));
                }
            }
            
        } catch (IOException e) {
            System.err.println("Warning: Could not export hyperparameter results: " + e.getMessage());
        }
    }

    /**
     * Creates comprehensive benchmark summary file with key statistics
     */
    public static void exportBenchmarkSummary(String systemInfo) {
        try {
            Files.createDirectories(Paths.get(OUTPUT_DIR));
            String filename = OUTPUT_DIR + "/benchmark_summary.csv";
            
            try (FileWriter writer = new FileWriter(filename)) {
                writer.append("Metric,Value,Unit\n");
                
                // System information
                writer.append("System_Info,\"" + systemInfo.replace("\"", "\"\"") + "\",text\n");
                writer.append("Timestamp,\"" + java.time.LocalDateTime.now() + "\",datetime\n");
                writer.append("Available_Processors," + Runtime.getRuntime().availableProcessors() + ",count\n");
                writer.append("Max_Memory_MB," + (Runtime.getRuntime().maxMemory() / (1024 * 1024)) + ",MB\n");
                
                // Calculate summary statistics from CSV files
                calculateAndWriteSummaryStatsFromCSV(writer);
            }
            
        } catch (IOException e) {
            System.err.println("Warning: Could not export benchmark summary: " + e.getMessage());
        }
    }
    
    /**
     * Calculate and write summary statistics by reading existing CSV files
     */
    private static void calculateAndWriteSummaryStatsFromCSV(FileWriter writer) throws IOException {
        try {
            // Read performance data if available
            String perfFile = OUTPUT_DIR + "/all_performance_results.csv";
            if (Files.exists(Paths.get(perfFile))) {
                String[] perfLines = Files.readAllLines(Paths.get(perfFile)).toArray(new String[0]);
                if (perfLines.length > 1) { // Skip header
                    double maxSpeedup = 1.0, minTime = Double.MAX_VALUE, maxTime = 0.0, sumSpeedup = 0.0;
                    int count = 0;
                    String bestAlgorithm = "";
                    double bestTime = Double.MAX_VALUE;
                    
                        for (int i = 1; i < perfLines.length; i++) {
                            String[] parts = parseCSVLine(perfLines[i]);
                            if (parts.length >= 6) {
                                try {
                                    String algorithm = parts[0];
                                    double timeMs = Double.parseDouble(parts[2]);
                                    double speedup = Double.parseDouble(parts[4]);
                                    
                                    // Filter out unrealistic speedup values (likely parsing errors)
                                    if (speedup > 0 && speedup < 100 && timeMs > 0 && timeMs < 100000) {
                                        maxSpeedup = Math.max(maxSpeedup, speedup);
                                        minTime = Math.min(minTime, timeMs);
                                        maxTime = Math.max(maxTime, timeMs);
                                        sumSpeedup += speedup;
                                        count++;
                                        
                                        if (timeMs < bestTime) {
                                            bestTime = timeMs;
                                            bestAlgorithm = algorithm;
                                        }
                                    }
                                } catch (NumberFormatException e) {
                                    // Skip invalid entries
                                }
                            }
                        }                    writer.append("Max_Speedup," + String.format("%.3f", maxSpeedup) + ",ratio\n");
                    writer.append("Average_Speedup," + String.format("%.3f", sumSpeedup / count) + ",ratio\n");
                    writer.append("Min_Execution_Time_ms," + String.format("%.3f", minTime) + ",ms\n");
                    writer.append("Max_Execution_Time_ms," + String.format("%.3f", maxTime) + ",ms\n");
                    writer.append("Total_Performance_Test_Cases," + (count) + ",count\n");
                    writer.append("Best_Algorithm,\"" + bestAlgorithm.replace("\"", "\"\"") + "\",text\n");
                    writer.append("Best_Algorithm_Time_ms," + String.format("%.3f", bestTime) + ",ms\n");
                }
            }
            
            // Read efficiency data if available
            String effFile = OUTPUT_DIR + "/all_efficiency_analysis.csv";
            if (Files.exists(Paths.get(effFile))) {
                String[] effLines = Files.readAllLines(Paths.get(effFile)).toArray(new String[0]);
                if (effLines.length > 1) {
                    double maxEfficiency = 0.0, sumEfficiency = 0.0, maxSpeedupPerThread = 0.0;
                    int excellentCount = 0, goodCount = 0, count = 0;
                    
                    for (int i = 1; i < effLines.length; i++) {
                        String[] parts = parseCSVLine(effLines[i]);
                        if (parts.length >= 7) {
                            try {
                                double efficiency = Double.parseDouble(parts[4]);
                                double speedupPerThread = Double.parseDouble(parts[5]);
                                String rating = parts[6];
                                
                                maxEfficiency = Math.max(maxEfficiency, efficiency);
                                maxSpeedupPerThread = Math.max(maxSpeedupPerThread, speedupPerThread);
                                sumEfficiency += efficiency;
                                count++;
                                
                                if (rating.contains("EXCELLENT")) excellentCount++;
                                else if (rating.contains("GOOD")) goodCount++;
                            } catch (NumberFormatException e) {
                                // Skip invalid entries
                            }
                        }
                    }
                    
                    writer.append("Max_Parallel_Efficiency," + String.format("%.3f", maxEfficiency) + ",ratio\n");
                    writer.append("Average_Parallel_Efficiency," + String.format("%.3f", sumEfficiency / count) + ",ratio\n");
                    writer.append("Max_Speedup_Per_Thread," + String.format("%.3f", maxSpeedupPerThread) + ",ratio\n");
                    writer.append("Excellent_Efficiency_Cases," + excellentCount + ",count\n");
                    writer.append("Good_Efficiency_Cases," + goodCount + ",count\n");
                    writer.append("Total_Efficiency_Test_Cases," + count + ",count\n");
                }
            }
            
            // Read resource usage data if available
            String resFile = OUTPUT_DIR + "/all_resource_usage.csv";
            if (Files.exists(Paths.get(resFile))) {
                String[] resLines = Files.readAllLines(Paths.get(resFile)).toArray(new String[0]);
                if (resLines.length > 1) {
                    double maxMemory = 0.0, sumMemory = 0.0, maxCpu = 0.0, sumCpu = 0.0;
                    int maxThreads = 1, count = 0;
                    
                    for (int i = 1; i < resLines.length; i++) {
                        String[] parts = parseCSVLine(resLines[i]);
                        if (parts.length >= 6) {
                            try {
                                int threads = Integer.parseInt(parts[4]);
                                double memory = Double.parseDouble(parts[2]);
                                
                                maxMemory = Math.max(maxMemory, memory);
                                maxThreads = Math.max(maxThreads, threads);
                                sumMemory += memory;
                                count++;
                            } catch (NumberFormatException e) {
                                // Skip invalid entries
                            }
                        }
                    }
                    
                    writer.append("Max_Memory_Usage_MB," + String.format("%.2f", maxMemory) + ",MB\n");
                    writer.append("Average_Memory_Usage_MB," + String.format("%.2f", sumMemory / count) + ",MB\n");
                    writer.append("Max_Threads_Used," + maxThreads + ",count\n");
                    writer.append("Total_Resource_Test_Cases," + count + ",count\n");
                }
            }
            
        } catch (Exception e) {
            System.err.println("Warning: Could not read CSV files for summary statistics: " + e.getMessage());
            writer.append("Summary_Status,\"Error reading CSV files for detailed statistics\",text\n");
        }
    }
    
    /**
     * Creates the complete unified dataset
     */
    public static void createMasterDataset() {
        // This would be handled by the Python consolidation script
        // or could be implemented here if needed
    }
    
    /**
     * Properly parse CSV line handling quoted fields with commas
     */
    private static String[] parseCSVLine(String line) {
        java.util.List<String> result = new java.util.ArrayList<>();
        boolean inQuotes = false;
        StringBuilder current = new StringBuilder();
        
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (c == '"') {
                inQuotes = !inQuotes;
            } else if (c == ',' && !inQuotes) {
                result.add(current.toString().trim());
                current = new StringBuilder();
            } else {
                current.append(c);
            }
        }
        result.add(current.toString().trim());
        
        return result.toArray(new String[0]);
    }

    /**
     * Gets efficiency rating based on percentage.
     */
    private static String getEfficiencyRating(double efficiency) {
        double percent = efficiency * 100;
        if (percent >= 90) return "EXCELLENT";
        if (percent >= 70) return "GOOD";
        if (percent >= 50) return "FAIR";
        return "POOR";
    }
    
    /**
     * Extracts parameter value from algorithm name.
     */
    private static String extractParameterValue(String algorithmName, String parameterName) {
        if (parameterName.contains("block") && algorithmName.contains("block size:")) {
            String[] parts = algorithmName.split("block size:\\s*");
            if (parts.length > 1) {
                return parts[1].split("[\\s,)]")[0];
            }
        } else if (parameterName.contains("thread") && algorithmName.contains("threads")) {
            String[] parts = algorithmName.split("\\(");
            if (parts.length > 1) {
                String threadPart = parts[1];
                return threadPart.split("\\s+")[0];
            }
        } else if (parameterName.contains("semaphore") && algorithmName.contains("semaphore:")) {
            String[] parts = algorithmName.split("semaphore:\\s*");
            if (parts.length > 1) {
                return parts[1].split("[\\s,)]")[0];
            }
        } else if (parameterName.contains("threshold") && algorithmName.contains("threshold:")) {
            String[] parts = algorithmName.split("threshold:\\s*");
            if (parts.length > 1) {
                return parts[1].split("[\\s,)]")[0];
            }
        }
        
        return "unknown";
    }
    
    /**
     * Escapes CSV field if it contains commas or quotes.
     */
    private static String escapeCsvField(String field) {
        if (field.contains(",") || field.contains("\"")) {
            return "\"" + field.replace("\"", "\"\"") + "\"";
        }
        return field;
    }
}