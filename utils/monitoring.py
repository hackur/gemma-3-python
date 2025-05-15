"""
System monitoring utilities for Gemma3 API Server.

This module provides classes and functions for monitoring system resources
and API performance metrics. It tracks CPU, memory, disk usage, and API
request statistics.

Classes:
    SystemMonitor: Monitors system resource usage
    PerformanceMonitor: Tracks API performance metrics

Usage:
    monitor = SystemMonitor()
    metrics = monitor.get_all_metrics()
"""

import psutil
import time
from typing import Dict, Any
from loguru import logger

class SystemMonitor:
    """
    Monitor system resources and performance.
    
    This class provides methods to collect and track system resource usage,
    including CPU, memory, disk, and process-specific metrics.
    """
    
    @staticmethod
    def get_cpu_info() -> Dict[str, Any]:
        """
        Get CPU usage information.
        
        Returns:
            Dict containing CPU usage percentage and count
            
        Example:
            {
                "cpu": {
                    "percent": 45.2,
                    "count": 8
                }
            }
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=0.5)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            cpu_stats = psutil.cpu_stats()
            
            return {
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count,
                    "freq": {
                        "current": cpu_freq.current if cpu_freq else 0,
                        "min": cpu_freq.min if cpu_freq else 0,
                        "max": cpu_freq.max if cpu_freq else 0
                    },
                    "stats": {
                        "ctx_switches": cpu_stats.ctx_switches,
                        "interrupts": cpu_stats.interrupts,
                        "soft_interrupts": cpu_stats.soft_interrupts,
                        "syscalls": cpu_stats.syscalls
                    }
                }
            }
        except Exception as e:
            logger.error(f"Error getting CPU info: {e}")
            return {"error": str(e)}

    @staticmethod
    def get_memory_info() -> Dict[str, Any]:
        """
        Get memory usage information.
        
        Returns:
            Dict containing memory usage statistics (total, available, used, free, percent)
            
        Example:
            {
                "memory": {
                    "total": 16777216,
                    "available": 8388608,
                    "used": 8388608,
                    "free": 8388608,
                    "percent": 50.0
                }
            }
        """
        try:
            mem = psutil.virtual_memory()
            return {
                "memory": {
                    "total": mem.total,
                    "available": mem.available,
                    "percent": mem.percent,
                    "used": mem.used,
                    "free": mem.free
                }
            }
        except Exception as e:
            logger.error(f"Error getting memory info: {e}")
            return {"error": str(e)}

    @staticmethod
    def get_disk_info() -> Dict[str, Any]:
        """
        Get disk usage information.
        
        Returns:
            Dict containing disk usage statistics (total, used, free, percent)
            
        Example:
            {
                "disk": {
                    "total": 1000000000,
                    "used": 500000000,
                    "free": 500000000,
                    "percent": 50.0
                }
            }
        """
        try:
            disk = psutil.disk_usage('/')
            return {
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": disk.percent
                }
            }
        except Exception as e:
            logger.error(f"Error getting disk info: {e}")
            return {"error": str(e)}

    @staticmethod
    def get_process_info() -> Dict[str, Any]:
        """
        Get current process information.
        
        Returns:
            Dict containing process metrics (CPU usage, memory usage, threads, open files, connections)
            
        Example:
            {
                "process": {
                    "cpu_percent": 2.5,
                    "memory_percent": 1.2,
                    "threads": 4,
                    "open_files": 12,
                    "connections": 2
                }
            }
        """
        try:
            process = psutil.Process()
            return {
                "process": {
                    "cpu_percent": process.cpu_percent(),
                    "memory_percent": process.memory_percent(),
                    "threads": process.num_threads(),
                    "open_files": len(process.open_files()),
                    "connections": len(process.connections())
                }
            }
        except Exception as e:
            logger.error(f"Error getting process info: {e}")
            return {"error": str(e)}

    @classmethod
    def get_all_metrics(cls) -> Dict[str, Any]:
        """
        Get all system metrics in a single call.
        
        Returns:
            Dict containing all available system metrics
            
        Example:
            {
                "cpu": {...},
                "memory": {...},
                "disk": {...},
                "process": {...}
            }
        """
        metrics = {}
        metrics.update(cls.get_cpu_info())
        metrics.update(cls.get_memory_info())
        metrics.update(cls.get_disk_info())
        metrics.update(cls.get_process_info())
        return metrics

class PerformanceMonitor:
    """
    Monitor API performance metrics.
    
    This class tracks request counts, error rates, and latency statistics
    for the API server.
    
    Attributes:
        start_time (float): Server start timestamp
        request_count (int): Total number of requests handled
        error_count (int): Total number of errors encountered
        total_latency (float): Cumulative request latency
    """
    
    def __init__(self):
        """Initialize performance monitoring."""
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.total_latency = 0
        
    def record_request(self, latency: float, is_error: bool = False):
        """
        Record metrics for a single request.
        
        Args:
            latency: Request processing time in seconds
            is_error: Whether the request resulted in an error
        """
        self.request_count += 1
        self.total_latency += latency
        if is_error:
            self.error_count += 1
            
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.
        
        Returns:
            Dict containing performance statistics
            
        Example:
            {
                "uptime_seconds": 3600,
                "total_requests": 1000,
                "error_rate": 0.01,
                "avg_latency": 0.15
            }
        """
        uptime = time.time() - self.start_time
        avg_latency = self.total_latency / self.request_count if self.request_count > 0 else 0
        
        return {
            "uptime_seconds": uptime,
            "total_requests": self.request_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "avg_latency": self.total_latency / max(self.request_count, 1)
        }