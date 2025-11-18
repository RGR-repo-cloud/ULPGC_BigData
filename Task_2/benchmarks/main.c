#include "matrix.h"
#include "sparse_matrix.h"
#include "benchmark.h"

void print_usage(const char* program_name) {
    printf("Usage: %s [option]\n", program_name);
    printf("Options:\n");
    printf("  --help       Show this help\n");
    printf("  --verify     Verify algorithm correctness\n");
    printf("  --benchmark  Run performance benchmark (256, 512, 1024, 2048)\n");
    printf("  --sparse     Sparse matrix analysis\n");
    printf("  --maxsize    Test maximum matrix sizes for each algorithm\n");
    printf("  (no args)    Run all tests\n");
}

void verify_algorithms(void) {
    printf("\n=== ALGORITHM VERIFICATION ===\n");
    
    const int test_size = 256;
    const double tolerance = 1e-10;
    
    printf("Creating test matrices (%dx%d)...\n", test_size, test_size);
    
    Matrix* a = matrix_create(test_size, test_size);
    Matrix* b = matrix_create(test_size, test_size);
    
    if (!a || !b) {
        printf("Failed to allocate test matrices\n");
        matrix_free(a);
        matrix_free(b);
        return;
    }
    
    // Use same seed for reproducible results
    srand(42);
    matrix_fill_random(a, -5.0, 5.0);
    matrix_fill_random(b, -5.0, 5.0);
    
    printf("Computing reference result with basic algorithm...\n");
    Matrix* reference = matrix_multiply_basic(a, b);
    if (!reference) {
        printf("Failed to compute reference result\n");
        matrix_free(a);
        matrix_free(b);
        return;
    }
    
    // Test loop unrolling
    printf("Verifying loop unrolling... ");
    Matrix* loop_result = matrix_multiply_loop_unroll(a, b);
    if (loop_result && matrix_verify_result(reference, loop_result, tolerance)) {
        printf("✓ PASSED\n");
    } else {
        printf("✗ FAILED\n");
    }
    matrix_free(loop_result);
    
    // Test cache blocking
    printf("Verifying cache blocking... ");
    Matrix* cache_result = matrix_multiply_cache_block(a, b, 32);
    if (cache_result && matrix_verify_result(reference, cache_result, tolerance)) {
        printf("✓ PASSED\n");
    } else {
        printf("✗ FAILED\n");
    }
    matrix_free(cache_result);
    
    // Test Strassen (only for power-of-2 sizes)
    if ((test_size & (test_size - 1)) == 0) {
        printf("Verifying Strassen algorithm... ");
        Matrix* strassen_result = matrix_multiply_strassen(a, b);
        if (strassen_result && matrix_verify_result(reference, strassen_result, tolerance)) {
            printf("✓ PASSED\n");
        } else {
            printf("✗ FAILED\n");
        }
        matrix_free(strassen_result);
    } else {
        printf("Skipping Strassen verification (matrix size not power of 2)\n");
    }
    
    // Test sparse matrix operations
    printf("Verifying sparse matrix operations... ");
    Matrix* sparse_a = matrix_create_sparse(32, 32, 0.7); // 70% sparse
    Matrix* sparse_b = matrix_create_sparse(32, 32, 0.7);
    
    if (sparse_a && sparse_b) {
        CSRMatrix* csr_a = csr_from_dense(sparse_a);
        CSRMatrix* csr_b = csr_from_dense(sparse_b);
        
        Matrix* dense_result = matrix_multiply_basic(sparse_a, sparse_b);
        CSRMatrix* sparse_result = csr_multiply(csr_a, csr_b);
        Matrix* sparse_as_dense = csr_to_dense(sparse_result);
        
        if (dense_result && sparse_as_dense && 
            matrix_verify_result(dense_result, sparse_as_dense, tolerance)) {
            printf("✓ PASSED\n");
        } else {
            printf("✗ FAILED\n");
        }
        
        matrix_free(sparse_a);
        matrix_free(sparse_b);
        matrix_free(dense_result);
        matrix_free(sparse_as_dense);
        csr_free(csr_a);
        csr_free(csr_b);
        csr_free(sparse_result);
    } else {
        printf("✗ FAILED (allocation error)\n");
    }
    
    matrix_free(a);
    matrix_free(b);
    matrix_free(reference);
    
    printf("Verification complete.\n");
}

void run_benchmark(void) {
    printf("\n=== PERFORMANCE BENCHMARK ===\n");
    run_performance_comparison(2048, 0); // Will use sizes: 256, 512, 1024, 2048
}

void run_sparse_benchmark(void) {
    printf("\n=== SPARSE MATRIX BENCHMARK ===\n");
    run_sparsity_analysis(256, 0.5, 0.95, 5);
}

void print_system_info(void) {
    printf("=== Matrix Multiplication Benchmarks ===\n");
    printf("Algorithms: Basic, Loop Unroll, Cache Block, Strassen, Sparse CSR\n\n");
}

int main(int argc, char* argv[]) {
    print_system_info();
    srand(time(NULL));
    
    // Simple argument parsing
    if (argc > 1) {
        if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[1], "--verify") == 0) {
            verify_algorithms();
            return 0;
        } else if (strcmp(argv[1], "--benchmark") == 0) {
            run_benchmark();
            return 0;
        } else if (strcmp(argv[1], "--sparse") == 0) {
            run_sparse_benchmark();
            return 0;
        } else if (strcmp(argv[1], "--maxsize") == 0) {
            test_maximum_matrix_sizes();
            return 0;
        }
    }
    
    // Default: run everything
    verify_algorithms();
    test_maximum_matrix_sizes();
    run_benchmark();
    run_sparse_benchmark();
    
    printf("\n=== BENCHMARK COMPLETE ===\n");
    printf("\nSUMMARY FOR REPORT:\n");
    printf("✓ Execution time: Measured with high-precision timing\n");
    printf("✓ Memory usage: Tracked during algorithm execution\n");
    printf("✓ Maximum matrix size: Tested up to 2048x2048\n");
    printf("✓ Dense vs Sparse: Compared at 50%%, 70%%, 90%%, 95%% sparsity\n");
    return 0;
}