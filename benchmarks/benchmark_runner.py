"""
Benchmark suite for Gemma3 API Server.

This module provides comprehensive benchmarking tools for measuring and analyzing
the performance characteristics of the Gemma3 API Server. It includes benchmarks
for model inference, memory usage, function calling, and system resource utilization.

Key Features:
    - Model inference latency measurement
    - Memory usage analysis and profiling
    - Function calling overhead measurement
    - Response streaming performance testing
    - System resource monitoring
    - Configurable iteration count
    - JSON results export

Example:
    >>> from benchmarks.benchmark_runner import Gemma3Benchmark
    >>> benchmark = Gemma3Benchmark(iterations=100)
    >>> results = benchmark.run_all()
    >>> print(json.dumps(results, indent=2))

Author: AI Developer
Version: 0.2.0
"""

import time
import psutil
import statistics
from pathlib import Path
import sys
import json
from typing import Dict, Any, List
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent))
from gemma3 import model, ChatMessage, ChatCompletionRequest

class Gemma3Benchmark:
    """
    Comprehensive benchmarking suite for Gemma3 API Server.
    
    This class provides methods to measure and analyze various performance aspects
    of the Gemma3 API Server. It includes benchmarks for model inference speed,
    memory usage patterns, function calling overhead, and system resource utilization.
    
    Attributes:
        iterations (int): Number of iterations for each benchmark
        model_path (str): Path to the model file
        results (Dict): Storage for benchmark results
        
    Args:
        iterations (int, optional): Number of iterations for each benchmark. Defaults to 100.
        model_path (str, optional): Path to model file. If None, uses default model.
        
    Example:
        >>> benchmark = Gemma3Benchmark(iterations=50)
        >>> results = benchmark.run_all()
        >>> benchmark.save_results("custom_results.json")
    """
    
    def __init__(self, iterations: int = 100, model_path: str = None):
        """Initialize benchmark suite with configuration."""
        self.iterations = iterations
        self.model_path = model_path
        self.results = {}
        
    def benchmark_inference(self) -> Dict[str, Any]:
        """
        Benchmark model inference performance.
        
        Returns:
            Dict containing latency statistics and memory usage metrics
        """
        logger.info(f"Starting inference benchmark ({self.iterations} iterations)")
        latencies = []
        memory_usage = []
        
        for i in range(self.iterations):
            if i % max(1, self.iterations // 10) == 0:  # Progress every 10%
                logger.info(f"Progress: {i}/{self.iterations} iterations ({i/self.iterations*100:.1f}%)")
            
            start_time = time.perf_counter()
            # Record memory before inference
            mem_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Perform inference
            response = self._run_inference()
            
            # Record metrics
            end_time = time.perf_counter()
            mem_after = psutil.Process().memory_info().rss / 1024 / 1024
            
            latencies.append(end_time - start_time)
            memory_usage.append(mem_after - mem_before)
            
            # Optional cooldown between iterations
            if i < self.iterations - 1:
                time.sleep(0.1)  # Prevent overheating/throttling
        
        logger.info("Benchmark complete, calculating statistics...")
        
        return {
            "latency": {
                "mean": statistics.mean(latencies),
                "median": statistics.median(latencies),
                "std_dev": statistics.stdev(latencies),
                "min": min(latencies),
                "max": max(latencies),
                "p95": sorted(latencies)[int(len(latencies) * 0.95)],
            },
            "memory": {
                "mean_delta_mb": statistics.mean(memory_usage),
                "peak_delta_mb": max(memory_usage)
            },
            "iterations": self.iterations
        }
        
    def benchmark_function_calling(self) -> Dict[str, Any]:
        """
        Benchmark function calling performance.
        
        Tests the overhead and performance characteristics of the function
        calling implementation. Measures parsing time, execution overhead,
        and response formatting time.
        
        Measures:
            - Function parsing latency
            - Execution overhead timing
            - Response formatting duration
            - Total function call latency
            
        Returns:
            Dict[str, Any]: Dictionary containing benchmark results with keys:
                - avg_total_latency: Mean total function call time
                - avg_parse_time: Mean time to parse function calls
                - avg_execution_time: Mean function execution time
                - p95_latency: 95th percentile total latency
        """
        function_latencies = []
        parsing_times = []
        execution_times = []
        
        test_message = {
            "role": "user",
            "content": "What is the current CPU usage?"
        }
        
        test_function = {
            "name": "get_system_info",
            "description": "Get system information",
            "parameters": {
                "type": "object",
                "properties": {
                    "info_type": {
                        "type": "string",
                        "enum": ["cpu", "memory", "disk", "all"]
                    }
                }
            }
        }
        
        for _ in range(self.iterations):
            start_time = time.time()
            completion = model.create_completion(
                prompt=str(test_message),
                functions=[test_function]
            )
            total_time = time.time() - start_time
            
            function_latencies.append(total_time)
            # Additional timing details captured in the completion
            if "timing" in completion:
                parsing_times.append(completion["timing"].get("parse_time", 0))
                execution_times.append(completion["timing"].get("execute_time", 0))
                
        return {
            "avg_total_latency": statistics.mean(function_latencies),
            "avg_parse_time": statistics.mean(parsing_times) if parsing_times else None,
            "avg_execution_time": statistics.mean(execution_times) if execution_times else None,
            "p95_latency": statistics.quantiles(function_latencies, n=20)[18]
        }
        
    def benchmark_memory(self) -> Dict[str, Any]:
        """
        Benchmark memory usage patterns.
        
        Analyzes memory usage during various operations including model loading,
        inference, function calling, and response streaming.
        
        Measures:
            - Baseline memory usage
            - Peak memory consumption
            - Memory growth patterns
            - Allocation/deallocation behavior
            
        Returns:
            Dict[str, Any]: Dictionary containing memory usage statistics with keys:
                - avg_baseline_mb: Mean baseline memory usage in MB
                - avg_peak_mb: Mean peak memory usage in MB
                - avg_increase_mb: Mean memory increase during operations
                - max_peak_mb: Maximum observed peak memory usage
        """
        memory_metrics = []
        
        for _ in range(self.iterations):
            process = psutil.Process()
            
            # Measure baseline
            baseline = process.memory_info().rss
            
            # Run test operations
            model.create_completion(
                prompt="Test prompt for memory measurement",
                max_tokens=100
            )
            
            # Measure peak
            peak = process.memory_info().rss
            
            # Calculate difference
            memory_metrics.append({
                "baseline_mb": baseline / (1024 * 1024),
                "peak_mb": peak / (1024 * 1024),
                "difference_mb": (peak - baseline) / (1024 * 1024)
            })
            
        return {
            "avg_baseline_mb": statistics.mean([m["baseline_mb"] for m in memory_metrics]),
            "avg_peak_mb": statistics.mean([m["peak_mb"] for m in memory_metrics]),
            "avg_increase_mb": statistics.mean([m["difference_mb"] for m in memory_metrics]),
            "max_peak_mb": max([m["peak_mb"] for m in memory_metrics])
        }
        
    def run_all(self) -> Dict[str, Any]:
        """
        Run all benchmarks and collect results.
        
        Executes all benchmark suites sequentially and aggregates results into
        a single comprehensive report. This is the main entry point for running
        the complete benchmark suite.
        
        Returns:
            Dict[str, Any]: Dictionary containing all benchmark results with keys:
                - inference: Results from inference benchmarks
                - function_calling: Results from function calling benchmarks
                - memory: Results from memory usage benchmarks
                
        Example:
            >>> benchmark = Gemma3Benchmark()
            >>> results = benchmark.run_all()
            >>> print(json.dumps(results, indent=2))
        """
        self.results = {
            "inference": self.benchmark_inference(),
            "function_calling": self.benchmark_function_calling(),
            "memory": self.benchmark_memory()
        }
        return self.results
        
    def save_results(self, output_path: str = "benchmark_results.json"):
        """
        Save benchmark results to a JSON file.
        
        Args:
            output_path (str, optional): Path to save results file.
                Defaults to "benchmark_results.json"
        """
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)

if __name__ == "__main__":
    logger.info("Starting Gemma3 benchmarks...")
    
    # Create benchmark instance with reduced iterations for testing
    benchmark = Gemma3Benchmark(iterations=10)
    results = benchmark.run_all()
    
    # Display results
    print("\nBenchmark Results:")
    print(json.dumps(results, indent=2))
    
    # Save results to file
    benchmark.save_results()
    logger.info("Benchmarks completed. Results saved to benchmark_results.json")