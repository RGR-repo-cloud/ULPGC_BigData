/**
 * Statistical Utilities for Java
 */
public class StatsUtils {
    
    /**
     * Calculate average of an array
     */
    public static double calculateAverage(double[] values) {
        double sum = 0;
        for (double value : values) {
            sum += value;
        }
        return sum / values.length;
    }
    
    /**
     * Find minimum value in array
     */
    public static double findMin(double[] values) {
        double min = values[0];
        for (double value : values) {
            if (value < min) min = value;
        }
        return min;
    }
    
    /**
     * Find maximum value in array
     */
    public static double findMax(double[] values) {
        double max = values[0];
        for (double value : values) {
            if (value > max) max = value;
        }
        return max;
    }
    
    /**
     * Calculate standard deviation
     */
    public static double calculateStdDev(double[] values) {
        if (values.length <= 1) return 0;
        
        double mean = calculateAverage(values);
        double sumSquaredDiffs = 0;
        
        for (double value : values) {
            double diff = value - mean;
            sumSquaredDiffs += diff * diff;
        }
        
        return Math.sqrt(sumSquaredDiffs / (values.length - 1));
    }
}