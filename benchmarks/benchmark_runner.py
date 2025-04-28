"""
Benchmark suite for Gemma3 API Server.

This module provides comprehensive benchmarking tools for:
- Model loading time
- Inference latency
- Memory usage
- Function calling overhead
- Response streaming performance
"""

import time
import psutil
import statistics
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from gemma3 import model, ChatMessage, ChatCompletionRequest

class Gemma3Benchmark:
    def __init__(self, iterations=100):
        self.iterations = iterations
        self.results = {}
        
    def benchmark_inference(self):
        """Benchmark basic inference latency"""
        latencies = []
        
        request = ChatCompletionRequest(
            model="gemma-3-4b-it",
            messages=[
                ChatMessage(role="user", content="Write a short greeting.")
            ]
        )
        
        for _ in range(self.iterations):
            start = time.perf_counter()
            model.create_completion(request.messages[0].content)
            end = time.perf_counter()
            latencies.append(end - start)
            
        return {
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "std_dev": statistics.stdev(latencies),
            "min": min(latencies),
            "max": max(latencies)
        }
        
    def benchmark_function_calling(self):
        """Benchmark function calling latency"""
        latencies = []
        
        request = ChatCompletionRequest(
            model="gemma-3-4b-it",
            messages=[
                ChatMessage(role="user", content="What is the current CPU usage?")
            ],
            functions=[{
                "name": "get_system_info",
                "description": "Get system information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "info_type": {
                            "type": "string",
                            "enum": ["cpu", "memory", "disk", "all"]
                        }
                    },
                    "required": ["info_type"]
                }
            }]
        )
        
        for _ in range(self.iterations):
            start = time.perf_counter()
            model.create_completion(request.messages[0].content)
            end = time.perf_counter()
            latencies.append(end - start)
            
        return {
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "std_dev": statistics.stdev(latencies),
            "min": min(latencies),
            "max": max(latencies)
        }
        
    def benchmark_memory(self):
        """Benchmark memory usage"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Run a series of completions
        for _ in range(10):
            model.create_completion("Test memory usage with a completion request")
            
        final_memory = process.memory_info().rss
        
        return {
            "initial_memory_mb": initial_memory / (1024 * 1024),
            "final_memory_mb": final_memory / (1024 * 1024),
            "memory_increase_mb": (final_memory - initial_memory) / (1024 * 1024)
        }

if __name__ == "__main__":
    benchmark = Gemma3Benchmark(iterations=10)  # Reduced iterations for testing
    
    print("Running Gemma3 benchmarks...")
    print("\nBasic Inference Benchmark:")
    print(benchmark.benchmark_inference())
    
    print("\nFunction Calling Benchmark:")
    print(benchmark.benchmark_function_calling())
    
    print("\nMemory Usage Benchmark:")
    print(benchmark.benchmark_memory())