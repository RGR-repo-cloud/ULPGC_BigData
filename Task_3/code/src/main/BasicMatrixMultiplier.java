package main;

/**
 * Basic sequential matrix multiplication implementation.
 * This serves as the baseline for performance comparison.
 */
public class BasicMatrixMultiplier implements MatrixMultiplier {
    
    @Override
    public double[][] multiply(double[][] A, double[][] B) {
        MatrixUtils.validateMultiplication(A, B);
        
        int rowsA = A.length;
        int colsA = A[0].length;
        int colsB = B[0].length;
        
        double[][] C = MatrixUtils.createResultMatrix(rowsA, colsB);
        
        // Traditional triple-nested loop matrix multiplication
        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsB; j++) {
                double sum = 0.0;
                for (int k = 0; k < colsA; k++) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }
        
        return C;
    }
    
    @Override
    public String getAlgorithmName() {
        return "Basic Sequential";
    }
}