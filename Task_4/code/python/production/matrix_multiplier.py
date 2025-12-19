import hazelcast
import numpy as np
import time
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistributedMatrixMultiplier:
    """
    Distributed matrix multiplication using Hazelcast In-Memory Data Grid.
    Implements block-wise matrix multiplication for scalability.
    """
    
    def __init__(self, cluster_name: str = "dev"):
        self.client = None
        self.cluster_name = cluster_name
        self.block_size = 64  # Default block size
        self.cluster_size = 0  # Number of nodes in cluster
        self.memory_per_node_mb = 0.0  # Estimated memory per node
        
    def connect(self) -> bool:
        """Connect to Hazelcast cluster."""
        try:
            self.client = hazelcast.HazelcastClient(
                cluster_name=self.cluster_name,
                cluster_members=["127.0.0.1:5701", "127.0.0.1:5702", "127.0.0.1:5703", "127.0.0.1:5704"]
            )
            # Get cluster size
            cluster = self.client.cluster_service
            self.cluster_size = len(cluster.get_members())
            logger.info(f"Connected to Hazelcast cluster with {self.cluster_size} node(s)")
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Hazelcast cluster."""
        if self.client:
            self.client.shutdown()
            logger.info("Disconnected from cluster")
    
    def _store_matrix_blocks(self, matrix: np.ndarray, name: str) -> List[Tuple[int, int]]:
        """Store matrix as distributed blocks."""
        rows, cols = matrix.shape
        block_indices = []
        
        # Get distributed map
        matrix_map = self.client.get_map(name).blocking()
        
        # Split matrix into blocks
        for i in range(0, rows, self.block_size):
            for j in range(0, cols, self.block_size):
                end_i = min(i + self.block_size, rows)
                end_j = min(j + self.block_size, cols)
                
                block = matrix[i:end_i, j:end_j]
                key = f"{i}_{j}"
                matrix_map.put(key, block.tolist())
                block_indices.append((i, j))
        
        return block_indices
    
    def _get_matrix_block(self, map_name: str, i: int, j: int) -> np.ndarray:
        """Retrieve matrix block from distributed storage."""
        matrix_map = self.client.get_map(map_name).blocking()
        key = f"{i}_{j}"
        block_data = matrix_map.get(key)
        return np.array(block_data) if block_data is not None else np.zeros((self.block_size, self.block_size))
    
    def multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Perform distributed matrix multiplication.
        
        Args:
            A: First matrix (m x n)
            B: Second matrix (n x p)
            
        Returns:
            Result matrix (m x p)
        """
        if A.shape[1] != B.shape[0]:
            raise ValueError("Matrix dimensions don't match for multiplication")
        
        m, n = A.shape
        n2, p = B.shape
        
        logger.info(f"Multiplying matrices: ({m}x{n}) * ({n2}x{p})")
        
        # Track network operations
        self.network_time = 0.0
        self.data_transferred_mb = 0.0
        
        # Store matrices in distributed storage
        logger.info("Storing matrices in distributed storage...")
        start_time = time.time()
        
        self._store_matrix_blocks(A, "matrix_A")
        self._store_matrix_blocks(B, "matrix_B")
        
        storage_time = time.time() - start_time
        self.network_time += storage_time
        
        # Calculate data transferred (upload)
        data_size_mb = (A.nbytes + B.nbytes) / (1024 * 1024)
        self.data_transferred_mb += data_size_mb
        
        # Estimate memory per node (data is distributed across cluster)
        self.memory_per_node_mb = data_size_mb / max(self.cluster_size, 1)
        
        logger.info(f"Storage time: {storage_time:.2f}s")
        logger.info(f"Data uploaded: {data_size_mb:.2f} MB")
        logger.info(f"Estimated memory per node: {self.memory_per_node_mb:.2f} MB")
        
        # Initialize result matrix
        C = np.zeros((m, p))
        
        # Perform block-wise multiplication
        logger.info("Performing distributed multiplication...")
        compute_start = time.time()
        
        blocks_retrieved = 0
        for i in range(0, m, self.block_size):
            for j in range(0, p, self.block_size):
                # Compute C[i:i+block_size, j:j+block_size]
                result_block = np.zeros((
                    min(self.block_size, m - i),
                    min(self.block_size, p - j)
                ))
                
                for k in range(0, n, self.block_size):
                    # Get blocks A[i:i+bs, k:k+bs] and B[k:k+bs, j:j+bs]
                    network_op_start = time.time()
                    block_A = self._get_matrix_block("matrix_A", i, k)
                    block_B = self._get_matrix_block("matrix_B", k, j)
                    self.network_time += (time.time() - network_op_start)
                    blocks_retrieved += 2
                    
                    # Adjust block sizes for edge cases
                    actual_A = block_A[:min(self.block_size, m-i), :min(self.block_size, n-k)]
                    actual_B = block_B[:min(self.block_size, n-k), :min(self.block_size, p-j)]
                    
                    # Use basic triple-loop multiplication (same as basic method)
                    if actual_A.size > 0 and actual_B.size > 0:
                        rows_A, cols_A = actual_A.shape
                        cols_B = actual_B.shape[1]
                        for ii in range(rows_A):
                            for jj in range(cols_B):
                                for kk in range(cols_A):
                                    result_block[ii, jj] += actual_A[ii, kk] * actual_B[kk, jj]
                
                # Store result in final matrix
                end_i = min(i + self.block_size, m)
                end_j = min(j + self.block_size, p)
                C[i:end_i, j:end_j] = result_block[:end_i-i, :end_j-j]
        
        compute_time = time.time() - compute_start
        
        # Add data transfer for block retrievals (matching Java implementation)
        block_data_mb = (blocks_retrieved * self.block_size * self.block_size * 8.0) / (1024 * 1024)
        self.data_transferred_mb += block_data_mb
        
        logger.info(f"Computation time: {compute_time:.2f}s")
        logger.info(f"Blocks retrieved: {blocks_retrieved}")
        logger.info(f"Total network time: {self.network_time:.2f}s")
        logger.info(f"Total data transferred: {self.data_transferred_mb:.2f} MB")
        
        # Cleanup distributed storage
        cleanup_start = time.time()
        self.client.get_map("matrix_A").blocking().clear()
        self.client.get_map("matrix_B").blocking().clear()
        cleanup_time = time.time() - cleanup_start
        self.network_time += cleanup_time
        
        return C


def basic_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Basic sequential matrix multiplication using triple loop."""
    m, n = A.shape
    n2, p = B.shape
    
    if n != n2:
        raise ValueError("Matrix dimensions don't match for multiplication")
    
    # Initialize result matrix
    C = np.zeros((m, p))
    
    # Triple loop matrix multiplication
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    
    return C


def _compute_rows_worker(args):
    """Worker function for parallel matrix multiplication."""
    start_row, end_row, A_chunk, B_full, p, n = args
    chunk_result = np.zeros((end_row - start_row, p))
    
    for i in range(end_row - start_row):
        for j in range(p):
            for k in range(n):
                chunk_result[i, j] += A_chunk[i, k] * B_full[k, j]
    
    return start_row, chunk_result


def parallel_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Parallel matrix multiplication using multiprocessing to bypass GIL."""
    from multiprocessing import Pool, cpu_count
    
    m, n = A.shape
    n2, p = B.shape
    
    if n != n2:
        raise ValueError("Matrix dimensions don't match for multiplication")
    
    # Determine number of processes (use CPU count)
    num_processes = cpu_count()
    
    # Calculate rows per process
    rows_per_process = m // num_processes
    
    # Prepare work chunks
    work_chunks = []
    for proc in range(num_processes):
        start_row = proc * rows_per_process
        end_row = (proc + 1) * rows_per_process if proc < num_processes - 1 else m
        A_chunk = A[start_row:end_row, :]
        work_chunks.append((start_row, end_row, A_chunk, B, p, n))
    
    # Use multiprocessing pool
    with Pool(processes=num_processes) as pool:
        results = pool.map(_compute_rows_worker, work_chunks)
    
    # Assemble results
    C = np.zeros((m, p))
    for start_row, chunk_result in results:
        end_row = start_row + chunk_result.shape[0]
        C[start_row:end_row, :] = chunk_result
    
    return C


if __name__ == "__main__":
    # Simple test
    multiplier = DistributedMatrixMultiplier()
    
    if not multiplier.connect():
        print("Failed to connect to Hazelcast. Make sure Hazelcast server is running.")
        exit(1)
    
    try:
        # Test with small matrices
        A = np.random.rand(100, 80)
        B = np.random.rand(80, 60)
        
        print("Testing distributed matrix multiplication...")
        start_time = time.time()
        result = multiplier.multiply(A, B)
        dist_time = time.time() - start_time
        
        print(f"Distributed multiplication completed in {dist_time:.2f}s")
        print(f"Result shape: {result.shape}")
        
        # Verify correctness
        expected = basic_multiply(A, B)
        if np.allclose(result, expected, rtol=1e-10):
            print("✓ Result is correct!")
        else:
            print("✗ Result verification failed!")
            
    finally:
        multiplier.disconnect()