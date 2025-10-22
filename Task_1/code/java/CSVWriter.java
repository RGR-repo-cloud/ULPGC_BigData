/**
 * CSV Writer for Java Matrix Multiplication Benchmark
 */
import java.io.*;

public class CSVWriter {
    
    /**
     * Write benchmark results to CSV file
     */
    public static void writeBenchmarkToCSV(String language, int[] sizes, 
            double[][] allTimes, double[][] allMemory, String csvFile) {
        
        if (csvFile == null) return;
        
        try (FileWriter fw = new FileWriter(csvFile, true);
             PrintWriter writer = new PrintWriter(fw)) {
            
            for (int s = 0; s < sizes.length; s++) {
                int size = sizes[s];
                double[] times = allTimes[s];
                double[] memory = allMemory[s];
                
                // Write each run as a separate row
                for (int run = 0; run < times.length; run++) {
                    writer.printf("%s,%d,%d,%.6f,%.2f%n",
                        language, size, run + 1, times[run], memory[run]);
                }
            }
            
        } catch (IOException e) {
            System.err.println("Warning: Could not write to CSV file: " + e.getMessage());
        }
    }
}