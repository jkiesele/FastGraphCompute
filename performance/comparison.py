"""
Performance comparison between FastGraphCompute (FGC) and FAISS for KNN search.
Compares speed, memory usage, and accuracy across different parameters.
"""

import os
import sys
import time
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import gc
import psutil
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import fastgraphcompute
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import torch.nn.functional as F
    from fastgraphcompute.extensions.binned_select_knn import binned_select_knn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS not available - install with: pip install faiss-gpu")

try:
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("scikit-learn not available")

@dataclass
class BenchmarkResult:
    """Store benchmark results for one configuration."""
    method: str
    k: int
    n_points: int
    dimensions: int
    runtime_ms: float
    memory_mb: float
    accuracy: float
    device: str
    
@dataclass
class TestConfig:
    """Configuration for benchmark tests."""
    k_values: List[int]
    n_points_list: List[int]
    dimensions_list: List[int]
    n_runs: int
    use_gpu: bool


class DataGenerator:
    """Generate synthetic datasets for benchmarking."""
    
    @staticmethod
    def generate_gaussian_clusters(n_points: int, dimensions: int, n_clusters: int = 5, 
                                 cluster_std: float = 1.0, random_state: int = 42) -> np.ndarray:
        """Generate data with Gaussian clusters."""
        np.random.seed(random_state)
        
        # Generate cluster centers
        centers = np.random.randn(n_clusters, dimensions) * 10
        
        # Generate points around centers
        points_per_cluster = n_points // n_clusters
        remainder = n_points % n_clusters
        
        data = []
        for i in range(n_clusters):
            n_cluster_points = points_per_cluster + (1 if i < remainder else 0)
            cluster_points = np.random.randn(n_cluster_points, dimensions) * cluster_std + centers[i]
            data.append(cluster_points)
        
        return np.vstack(data).astype(np.float32)
    
    @staticmethod
    def generate_uniform_random(n_points: int, dimensions: int, 
                               bounds: Tuple[float, float] = (-10, 10), 
                               random_state: int = 42) -> np.ndarray:
        """Generate uniformly random data."""
        np.random.seed(random_state)
        return np.random.uniform(bounds[0], bounds[1], (n_points, dimensions)).astype(np.float32)
    
    @staticmethod
    def generate_sphere_surface(n_points: int, dimensions: int, 
                               radius: float = 10.0, random_state: int = 42) -> np.ndarray:
        """Generate points on sphere surface."""
        np.random.seed(random_state)
        # Generate random direction vectors
        points = np.random.randn(n_points, dimensions)
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms * radius
        return points.astype(np.float32)


class KNNImplementations:
    """Wrapper for different KNN implementations."""
    
    @staticmethod
    def fgc_knn(coords: np.ndarray, k: int, use_gpu: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """FastGraphCompute KNN implementation."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        coords_tensor = torch.from_numpy(coords).to(device)
        
        # Create row_splits for single batch
        row_splits = torch.tensor([0, len(coords)], dtype=torch.int64, device=device)
        
        # Run FGC KNN
        if torch.cuda.is_available() and use_gpu:
            torch.cuda.synchronize()
        
        indices, distances = binned_select_knn(
            K=k,
            coords=coords_tensor,
            row_splits=row_splits,
            max_bin_dims=3
        )
        
        if torch.cuda.is_available() and use_gpu:
            torch.cuda.synchronize()
        
        return indices.cpu().numpy(), distances.cpu().numpy()
    
    @staticmethod
    def faiss_knn(coords: np.ndarray, k: int, use_gpu: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """FAISS KNN implementation."""
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available")
        
        dimensions = coords.shape[1]
        
        # Build index
        if use_gpu and faiss.get_num_gpus() > 0:
            # GPU version
            index = faiss.IndexFlatL2(dimensions)
            index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
        else:
            # CPU version
            index = faiss.IndexFlatL2(dimensions)
        
        # Add vectors to index
        index.add(coords)
        
        # Search
        distances, indices = index.search(coords, k)
        
        return indices, distances
    
    @staticmethod
    def sklearn_knn(coords: np.ndarray, k: int, use_gpu: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Scikit-learn KNN implementation."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not available")
        
        # sklearn doesn't support GPU
        nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=-1)
        nbrs.fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        
        return indices, distances
    
    @staticmethod
    def pytorch_naive_knn(coords: np.ndarray, k: int, use_gpu: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Naive PyTorch KNN implementation for ground truth."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        coords_tensor = torch.from_numpy(coords).to(device)
        
        # Compute pairwise distances
        dist_matrix = torch.cdist(coords_tensor, coords_tensor, p=2)
        
        # Get k nearest neighbors
        distances, indices = torch.topk(dist_matrix, k, dim=1, largest=False)
        
        return indices.cpu().numpy(), distances.cpu().numpy()


class PerformanceBenchmark:
    """Main benchmarking class."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.results: List[BenchmarkResult] = []
        
        # Check available implementations
        self.available_methods = {}
        if TORCH_AVAILABLE:
            self.available_methods['FGC'] = KNNImplementations.fgc_knn
            self.available_methods['PyTorch_Naive'] = KNNImplementations.pytorch_naive_knn
        if FAISS_AVAILABLE:
            self.available_methods['FAISS'] = KNNImplementations.faiss_knn
        if SKLEARN_AVAILABLE:
            self.available_methods['sklearn'] = KNNImplementations.sklearn_knn
        
        print(f"Available methods: {list(self.available_methods.keys())}")
    
    def measure_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def compute_accuracy(self, indices_test: np.ndarray, indices_ground_truth: np.ndarray, k: int) -> float:
        """Compute recall@k accuracy."""
        if indices_test.shape != indices_ground_truth.shape:
            return 0.0
        
        # For each query point, compute intersection with ground truth
        total_correct = 0
        total_possible = len(indices_test) * k
        
        for i in range(len(indices_test)):
            test_neighbors = set(indices_test[i])
            gt_neighbors = set(indices_ground_truth[i])
            correct = len(test_neighbors.intersection(gt_neighbors))
            total_correct += correct
        
        return total_correct / total_possible
    
    def benchmark_method(self, method_name: str, method_func: callable, 
                        coords: np.ndarray, k: int, n_runs: int) -> Dict[str, float]:
        """Benchmark a single method."""
        print(f"  Testing {method_name}...")
        
        runtimes = []
        memory_usage = []
        
        for run in range(n_runs):
            # Clear cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Measure initial memory
            mem_start = self.measure_memory_usage()
            
            # Run benchmark
            start_time = time.time()
            try:
                if method_name == 'sklearn':
                    indices, distances = method_func(coords, k, use_gpu=False)
                else:
                    indices, distances = method_func(coords, k, use_gpu=self.config.use_gpu)
                
                end_time = time.time()
                runtime = (end_time - start_time) * 1000  # Convert to ms
                
                # Measure peak memory
                mem_end = self.measure_memory_usage()
                memory_used = mem_end - mem_start
                
                runtimes.append(runtime)
                memory_usage.append(memory_used)
                
                # Store results for accuracy computation
                if run == 0:  # Store first run results
                    first_run_results = (indices, distances)
                    
            except Exception as e:
                print(f"    Error in {method_name}: {e}")
                return None
        
        return {
            'runtime_ms': np.mean(runtimes),
            'runtime_std': np.std(runtimes),
            'memory_mb': np.mean(memory_usage),
            'results': first_run_results
        }
    
    def run_benchmark(self, k: int, n_points: int, dimensions: int, 
                     data_type: str = 'gaussian') -> None:
        """Run benchmark for specific configuration."""
        print(f"\nBenchmarking K={k}, N={n_points}, D={dimensions}, data={data_type}")
        
        # Generate data
        if data_type == 'gaussian':
            coords = DataGenerator.generate_gaussian_clusters(n_points, dimensions)
        elif data_type == 'uniform':
            coords = DataGenerator.generate_uniform_random(n_points, dimensions)
        elif data_type == 'sphere':
            coords = DataGenerator.generate_sphere_surface(n_points, dimensions)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        # Compute ground truth (using naive PyTorch if available)
        ground_truth_indices = None
        if 'PyTorch_Naive' in self.available_methods:
            print("  Computing ground truth...")
            try:
                gt_results = self.benchmark_method('PyTorch_Naive', 
                                                 self.available_methods['PyTorch_Naive'],
                                                 coords, k, 1)
                if gt_results:
                    ground_truth_indices = gt_results['results'][0]
            except Exception as e:
                print(f"  Ground truth computation failed: {e}")
        
        # Benchmark each method
        for method_name, method_func in self.available_methods.items():
            # Skip naive PyTorch for actual benchmarking (too slow for large datasets)
            if method_name == 'PyTorch_Naive' and n_points > 1000:
                continue
            
            results = self.benchmark_method(method_name, method_func, coords, k, self.config.n_runs)
            
            if results:
                # Compute accuracy
                accuracy = 1.0  # Default
                if ground_truth_indices is not None and method_name != 'PyTorch_Naive':
                    accuracy = self.compute_accuracy(results['results'][0], ground_truth_indices, k)
                
                # Store results
                device = 'GPU' if self.config.use_gpu else 'CPU'
                result = BenchmarkResult(
                    method=method_name,
                    k=k,
                    n_points=n_points,
                    dimensions=dimensions,
                    runtime_ms=results['runtime_ms'],
                    memory_mb=results['memory_mb'],
                    accuracy=accuracy,
                    device=device
                )
                self.results.append(result)
                
                print(f"    {method_name}: {results['runtime_ms']:.2f}ms, "
                      f"{results['memory_mb']:.1f}MB, accuracy={accuracy:.3f}")
    
    def run_full_benchmark(self) -> None:
        """Run full benchmark suite."""
        print("Starting performance comparison...")
        print(f"Configuration: {self.config}")
        
        total_tests = len(self.config.k_values) * len(self.config.n_points_list) * len(self.config.dimensions_list)
        current_test = 0
        
        for k in self.config.k_values:
            for n_points in self.config.n_points_list:
                for dimensions in self.config.dimensions_list:
                    current_test += 1
                    print(f"\n[{current_test}/{total_tests}] Running benchmark...")
                    
                    try:
                        self.run_benchmark(k, n_points, dimensions, 'gaussian')
                    except Exception as e:
                        print(f"Error in benchmark: {e}")
                        continue
        
        print(f"\nCompleted {len(self.results)} benchmark runs.")
    
    def save_results_csv(self, filename: str = 'knn_performance_results.csv') -> None:
        """Save results to CSV file."""
        if not self.results:
            print("No results to save")
            return
        
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = ['method', 'k', 'n_points', 'dimensions', 'runtime_ms', 
                         'memory_mb', 'accuracy', 'device']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in self.results:
                writer.writerow({
                    'method': result.method,
                    'k': result.k,
                    'n_points': result.n_points,
                    'dimensions': result.dimensions,
                    'runtime_ms': result.runtime_ms,
                    'memory_mb': result.memory_mb,
                    'accuracy': result.accuracy,
                    'device': result.device
                })
        
        print(f"Results saved to {filepath}")
    
    def generate_plots(self) -> None:
        """Generate performance comparison plots."""
        if not self.results:
            print("No results to plot")
            return
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame([{
            'method': r.method,
            'k': r.k,
            'n_points': r.n_points,
            'dimensions': r.dimensions,
            'runtime_ms': r.runtime_ms,
            'memory_mb': r.memory_mb,
            'accuracy': r.accuracy,
            'device': r.device
        } for r in self.results])
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Runtime vs Dataset Size
        ax1 = axes[0, 0]
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            ax1.loglog(method_data['n_points'], method_data['runtime_ms'], 
                      marker='o', label=method, linewidth=2)
        ax1.set_xlabel('Number of Points')
        ax1.set_ylabel('Runtime (ms)')
        ax1.set_title('Runtime vs Dataset Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Runtime vs K
        ax2 = axes[0, 1]
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            avg_runtime = method_data.groupby('k')['runtime_ms'].mean()
            ax2.plot(avg_runtime.index, avg_runtime.values, 
                    marker='o', label=method, linewidth=2)
        ax2.set_xlabel('K (number of neighbors)')
        ax2.set_ylabel('Runtime (ms)')
        ax2.set_title('Runtime vs K')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Memory Usage
        ax3 = axes[1, 0]
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            ax3.loglog(method_data['n_points'], method_data['memory_mb'], 
                      marker='s', label=method, linewidth=2)
        ax3.set_xlabel('Number of Points')
        ax3.set_ylabel('Memory Usage (MB)')
        ax3.set_title('Memory Usage vs Dataset Size')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Accuracy Comparison
        ax4 = axes[1, 1]
        accuracy_data = df[df['method'] != 'PyTorch_Naive']  # Exclude ground truth
        if not accuracy_data.empty:
            method_accuracy = accuracy_data.groupby('method')['accuracy'].mean()
            bars = ax4.bar(method_accuracy.index, method_accuracy.values, 
                          color=['skyblue', 'lightgreen', 'lightcoral', 'gold'][:len(method_accuracy)])
            ax4.set_ylabel('Accuracy (Recall@K)')
            ax4.set_title('Average Accuracy by Method')
            ax4.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, method_accuracy.values):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(os.path.dirname(__file__), 'knn_performance_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Plots saved to {plot_path}")


def main():
    """Main function to run the benchmark."""
    
    # Start with small test configuration
    small_config = TestConfig(
        k_values=[8, 16, 32],
        n_points_list=[100, 500, 1000],
        dimensions_list=[2, 4, 8],
        n_runs=3,
        use_gpu=torch.cuda.is_available() if TORCH_AVAILABLE else False
    )
    
    print("=== Small Scale Performance Comparison ===")
    benchmark = PerformanceBenchmark(small_config)
    benchmark.run_full_benchmark()
    
    # Save results
    benchmark.save_results_csv('knn_performance_small.csv')
    benchmark.generate_plots()
    
    # Option to run larger tests
    print("\n=== Optional: Large Scale Tests ===")
    run_large = input("Run large scale tests? (y/n): ").lower().strip() == 'y'
    
    if run_large:
        large_config = TestConfig(
            k_values=[16, 32, 64, 128],
            n_points_list=[1000, 5000, 10000, 50000],
            dimensions_list=[3, 8, 16],
            n_runs=3,
            use_gpu=torch.cuda.is_available() if TORCH_AVAILABLE else False
        )
        
        print("Running large scale benchmark...")
        large_benchmark = PerformanceBenchmark(large_config)
        large_benchmark.run_full_benchmark()
        
        # Save results
        large_benchmark.save_results_csv('knn_performance_large.csv')
        large_benchmark.generate_plots()


if __name__ == "__main__":
    main()