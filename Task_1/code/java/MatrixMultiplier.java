/**
 * Matrix multiplication implementation in Java
 */
public class MatrixMultiplier {
    
    /**
     * Basic matrix multiplication algorithm O(n^3)
     */
    public static double[][] matrixMultiply(double[][] A, double[][] B) {
        int n = A.length;
        double[][] C = new double[n][n];
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        
        return C;
    }
}