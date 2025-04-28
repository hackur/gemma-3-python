"""
System monitoring utilities for Gemma3 API Server.

Provides functions for monitoring system resources and performance metrics.
"""

import psutil
import time
from typing import Dict, Any
from loguru import logger

class SystemMonitor:
    """Monitor system resources and performance."""
    
    @staticmethod
    def get_cpu_info() -> Dict[str, Any]:
        """Get CPU usage information."""
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
        """Get memory usage information."""
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
        """Get disk usage information."""
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
        """Get current process information."""
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
        """Get all system metrics."""
        metrics = {}
        metrics.update(cls.get_cpu_info())
        metrics.update(cls.get_memory_info())
        metrics.update(cls.get_disk_info())
        metrics.update(cls.get_process_info())
        return metrics

class PerformanceMonitor:
    """Monitor API performance metrics."""
    
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.total_latency = 0
        
    def record_request(self, latency: float, is_error: bool = False):
        """Record a request's metrics."""
        self.request_count += 1
        self.total_latency += latency
        if is_error:
            self.error_count += 1
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        uptime = time.time() - self.start_time
        avg_latency = self.total_latency / self.request_count if self.request_count > 0 else 0
        
        return {
            "uptime": uptime,
            "requests": {
                "total": self.request_count,
                "errors": self.error_count,
                "success_rate": (self.request_count - self.error_count) / self.request_count if self.request_count > 0 else 0
            },
            "latency": {
                "total": self.total_latency,
                "average": avg_latency
            }
        }