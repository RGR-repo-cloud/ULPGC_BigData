/**
 * Matrix Utilities for Java
 */
import java.util.Random;

public class MatrixUtils {
    
    /**
     * Create a random matrix of given size
     */
    public static double[][] createRandomMatrix(int size) {
        Random random = new Random();
        double[][] matrix = new double[size][size];
        
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                matrix[i][j] = random.nextDouble();
            }
        }
        
        return matrix;
    }
    
    /**
     * Create a zero matrix of given size
     */
    public static double[][] createZeroMatrix(int size) {
        return new double[size][size];
    }
}