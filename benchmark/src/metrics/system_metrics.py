"""
System-Level Performance Metrics Collection.

This module provides metrics collection for system-level performance and resource utilization,
focusing on throughput, latency, and resource consumption across the entire multi-agent system.
"""

from typing import Dict, List, Any, Optional, Set, Union
import time
from datetime import datetime
from dataclasses import dataclass, field
import threading
import psutil
import statistics
import numpy as np
from collections import defaultdict, deque

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from benchmark.src.metrics.collectors import BaseMetricsCollector, MetricsCollectionConfig
from benchmark.src.instrumentation.memory_instrumentation import MemoryInstrumenter


@dataclass
class SystemMetricsConfig(MetricsCollectionConfig):
    """Configuration for system metrics collection."""
    
    # Resource monitoring configuration
    monitor_cpu: bool = True
    monitor_memory: bool = True
    monitor_disk: bool = True
    monitor_network: bool = True
    monitor_gpu: bool = False
    
    # Process monitoring
    monitor_process_stats: bool = True
    process_ids_to_monitor: Set[int] = field(default_factory=set)
    
    # Throughput metrics configuration
    throughput_window_seconds: int = 60
    
    # Latency metrics configuration
    latency_percentiles: List[float] = field(default_factory=lambda: [50.0, 90.0, 95.0, 99.0, 99.9])
    
    # Queue monitoring
    monitor_queue_depths: bool = True
    queue_names_to_monitor: Set[str] = field(default_factory=set)
    
    # Hardware specification defaults for estimation
    default_gpu_flops: float = 20.0e12  # 20 TFLOPS for a mid-range GPU
    default_memory_bandwidth: float = 600.0e9  # 600 GB/s for a mid-range GPU
    default_tokens_per_second: int = 30  # Default tokens per second for generation


class SystemMetricsCollector(BaseMetricsCollector):
    """
    Collector for system-level performance metrics.
    
    Captures throughput, latency, and resource utilization metrics across the entire system.
    """
    
    def __init__(self, config: Optional[SystemMetricsConfig] = None):
        """
        Initialize the system metrics collector.
        
        Args:
            config: Configuration for system metrics collection
        """
        super().__init__(config or SystemMetricsConfig())
        self.config = config or SystemMetricsConfig()
        self._collection_thread = None
        self._stop_collection = False
        
        # Store task completions for throughput calculation
        self._task_completions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Store latency measurements for percentile calculation
        self._latency_measurements: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        
        # Store queue depths
        self._queue_depths: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Resource utilization caches
        self._last_net_io = None
        self._last_net_io_time = None
        
    def _collection_loop(self) -> None:
        """Background thread that collects resource metrics at regular intervals."""
        while not self._stop_collection:
            try:
                if self.config.monitor_cpu:
                    self._collect_cpu_metrics()
                
                if self.config.monitor_memory:
                    self._collect_memory_metrics()
                
                if self.config.monitor_disk:
                    self._collect_disk_metrics()
                
                if self.config.monitor_network:
                    self._collect_network_metrics()
                
                if self.config.monitor_gpu and GPU_AVAILABLE:
                    self._collect_gpu_metrics()
                
                # Sleep until next collection interval
                time.sleep(self.config.sampling_interval_ms / 1000)
            except Exception as e:
                print(f"Error in system metrics collection: {str(e)}")
                time.sleep(1)  # Sleep before retrying
    
    def _collect_cpu_metrics(self) -> None:
        """Collect CPU utilization metrics."""
        # System-wide CPU
        cpu_percent = psutil.cpu_percent(interval=None)
        self.collect_point("system.cpu.utilization", cpu_percent, {"unit": "percent"})
        
        # Per-core CPU
        per_cpu = psutil.cpu_percent(interval=None, percpu=True)
        for i, cpu in enumerate(per_cpu):
            self.collect_point("system.cpu.core.utilization", cpu, {"core": str(i), "unit": "percent"})
        
        # Process-specific CPU if configured
        if self.config.monitor_process_stats:
            process_ids = self.config.process_ids_to_monitor
            if not process_ids:
                # If no specific PIDs provided, monitor current process
                process_ids = {psutil.Process().pid}
            
            for pid in process_ids:
                try:
                    process = psutil.Process(pid)
                    process_cpu = process.cpu_percent(interval=None)
                    self.collect_point("system.process.cpu.utilization", process_cpu, 
                                      {"pid": str(pid), "unit": "percent"})
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
    
    def _collect_memory_metrics(self) -> None:
        """Collect memory utilization metrics."""
        # System memory
        memory = psutil.virtual_memory()
        self.collect_point("system.memory.total", memory.total, {"unit": "bytes"})
        self.collect_point("system.memory.available", memory.available, {"unit": "bytes"})
        self.collect_point("system.memory.used", memory.used, {"unit": "bytes"})
        self.collect_point("system.memory.percent", memory.percent, {"unit": "percent"})
        
        # Process-specific memory if configured
        if self.config.monitor_process_stats:
            process_ids = self.config.process_ids_to_monitor
            if not process_ids:
                # If no specific PIDs provided, monitor current process
                process_ids = {psutil.Process().pid}
            
            for pid in process_ids:
                try:
                    process = psutil.Process(pid)
                    memory_info = process.memory_info()
                    self.collect_point("system.process.memory.rss", memory_info.rss, 
                                      {"pid": str(pid), "unit": "bytes"})
                    self.collect_point("system.process.memory.vms", memory_info.vms, 
                                      {"pid": str(pid), "unit": "bytes"})
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
    
    def _collect_disk_metrics(self) -> None:
        """Collect disk utilization metrics."""
        # Disk usage for major mounts
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                tags = {"mountpoint": partition.mountpoint, "unit": "bytes"}
                
                self.collect_point("system.disk.total", usage.total, tags)
                self.collect_point("system.disk.used", usage.used, tags)
                self.collect_point("system.disk.free", usage.free, tags)
                self.collect_point("system.disk.percent", usage.percent, 
                                  {**tags, "unit": "percent"})
            except (PermissionError, FileNotFoundError):
                # Some mountpoints might not be accessible
                pass
        
        # Disk I/O stats
        disk_io = psutil.disk_io_counters()
        if disk_io:
            self.collect_point("system.disk.read_bytes", disk_io.read_bytes, {"unit": "bytes"})
            self.collect_point("system.disk.write_bytes", disk_io.write_bytes, {"unit": "bytes"})
            self.collect_point("system.disk.read_count", disk_io.read_count, {"unit": "count"})
            self.collect_point("system.disk.write_count", disk_io.write_count, {"unit": "count"})
    
    def _collect_network_metrics(self) -> None:
        """Collect network utilization metrics."""
        net_io = psutil.net_io_counters()
        
        # Store absolute counters
        self.collect_point("system.network.bytes_sent", net_io.bytes_sent, {"unit": "bytes"})
        self.collect_point("system.network.bytes_recv", net_io.bytes_recv, {"unit": "bytes"})
        self.collect_point("system.network.packets_sent", net_io.packets_sent, {"unit": "count"})
        self.collect_point("system.network.packets_recv", net_io.packets_recv, {"unit": "count"})
        
        # Calculate rates if we have previous measurements
        current_time = time.time()
        if self._last_net_io and self._last_net_io_time:
            time_diff = current_time - self._last_net_io_time
            
            if time_diff > 0:
                bytes_sent_rate = (net_io.bytes_sent - self._last_net_io.bytes_sent) / time_diff
                bytes_recv_rate = (net_io.bytes_recv - self._last_net_io.bytes_recv) / time_diff
                
                self.collect_point("system.network.bytes_sent_per_sec", bytes_sent_rate, 
                                  {"unit": "bytes_per_second"})
                self.collect_point("system.network.bytes_recv_per_sec", bytes_recv_rate, 
                                  {"unit": "bytes_per_second"})
        
        # Update last values for next calculation
        self._last_net_io = net_io
        self._last_net_io_time = current_time
    
    def _collect_gpu_metrics(self) -> None:
        """Collect GPU utilization metrics if available."""
        if not GPU_AVAILABLE:
            return
            
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                tags = {"gpu_id": str(i), "gpu_name": gpu.name}
                
                self.collect_point("system.gpu.utilization", gpu.load * 100, 
                                  {**tags, "unit": "percent"})
                self.collect_point("system.gpu.memory.used", gpu.memoryUsed, 
                                  {**tags, "unit": "MB"})
                self.collect_point("system.gpu.memory.total", gpu.memoryTotal, 
                                  {**tags, "unit": "MB"})
                self.collect_point("system.gpu.temperature", gpu.temperature, 
                                  {**tags, "unit": "celsius"})
        except Exception as e:
            print(f"Error collecting GPU metrics: {str(e)}")
    
    # Throughput Metrics
    
    def record_task_completion(self, task_type: str, duration_ms: float, tags: Dict[str, str] = None) -> None:
        """
        Record the completion of a task for throughput calculation.
        
        Args:
            task_type: Type of task completed
            duration_ms: Duration of the task in milliseconds
            tags: Additional tags for the task
        """
        tags = tags or {}
        completion_time = time.time()
        
        with self._lock:
            self._task_completions[task_type].append({
                'time': completion_time,
                'duration_ms': duration_ms,
                'tags': tags
            })
            
            # Collect as point for time series
            self.collect_point(f"system.task.completion.{task_type}", 1, tags)
            self.collect_point(f"system.task.duration.{task_type}", duration_ms, {**tags, "unit": "ms"})
            
            # Prune old completions
            window = self.config.throughput_window_seconds
            cutoff_time = completion_time - window
            self._task_completions[task_type] = [
                tc for tc in self._task_completions[task_type]
                if tc['time'] > cutoff_time
            ]
    
    def get_throughput(self, task_type: Optional[str] = None, 
                      window_seconds: Optional[int] = None) -> float:
        """
        Calculate current throughput for the specified task type.
        
        Args:
            task_type: Type of task to calculate throughput for, or None for all
            window_seconds: Window to calculate throughput over, or None for default
            
        Returns:
            Tasks per second over the specified window
        """
        window = window_seconds or self.config.throughput_window_seconds
        current_time = time.time()
        cutoff_time = current_time - window
        
        with self._lock:
            if task_type:
                # Calculate throughput for specific task type
                completions = [
                    tc for tc in self._task_completions.get(task_type, [])
                    if tc['time'] > cutoff_time
                ]
                completion_count = len(completions)
            else:
                # Calculate throughput across all task types
                completion_count = sum(
                    len([tc for tc in completions if tc['time'] > cutoff_time])
                    for completions in self._task_completions.values()
                )
            
            if window > 0:
                return completion_count / window
            return 0.0
    
    # Latency Metrics
    
    def record_latency(self, operation_name: str, latency_ms: float, 
                       tags: Dict[str, str] = None) -> None:
        """
        Record a latency measurement.
        
        Args:
            operation_name: Name of the operation
            latency_ms: Latency in milliseconds
            tags: Additional tags for the operation
        """
        tags = tags or {}
        
        with self._lock:
            # Store for percentile calculations
            self._latency_measurements[operation_name].append(latency_ms)
            
            # Collect as point for time series
            self.collect_point(f"system.latency.{operation_name}", latency_ms, {**tags, "unit": "ms"})
    
    def get_latency_percentiles(self, operation_name: str, 
                               percentiles: Optional[List[float]] = None) -> Dict[float, float]:
        """
        Get latency percentiles for the specified operation.
        
        Args:
            operation_name: Name of the operation to get percentiles for
            percentiles: List of percentiles to calculate, or None for defaults
            
        Returns:
            Dictionary mapping percentiles to latency values
        """
        percentiles = percentiles or self.config.latency_percentiles
        
        with self._lock:
            measurements = list(self._latency_measurements.get(operation_name, []))
            
            if not measurements:
                return {p: 0.0 for p in percentiles}
            
            result = {}
            for p in percentiles:
                try:
                    result[p] = float(np.percentile(measurements, p))
                except Exception:
                    # Fallback if numpy is not available
                    sorted_latencies = sorted(measurements)
                    idx = int(len(sorted_latencies) * p / 100)
                    result[p] = sorted_latencies[min(idx, len(sorted_latencies) - 1)]
            
            return result
    
    # Queue Metrics
    
    def record_queue_depth(self, queue_name: str, depth: int, 
                          tags: Dict[str, str] = None) -> None:
        """
        Record the current depth of a queue.
        
        Args:
            queue_name: Name of the queue
            depth: Current depth of the queue
            tags: Additional tags for the queue
        """
        tags = tags or {}
        
        # Only monitor queues we care about if filter is set
        if (self.config.queue_names_to_monitor and 
            queue_name not in self.config.queue_names_to_monitor):
            return
        
        # Collect queue depth as a regular metric point - no locking needed
        self.collect_point(f"system.queue.depth.{queue_name}", depth, tags)
    
    def get_queue_metrics(self, queue_name: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Get metrics for the specified queue.
        
        Args:
            queue_name: Optional name of queue to get metrics for, or None for all
            
        Returns:
            Dictionary mapping queue names to metric dictionaries
        """
        result = {}
        
        with self._lock:
            # Determine which queues to process
            queue_names = [queue_name] if queue_name else self._queue_depths.keys()
            
            for name in queue_names:
                if name not in self._queue_depths:
                    continue
                
                depths = [qd['depth'] for qd in self._queue_depths[name]]
                
                if not depths:
                    result[name] = {
                        'current': 0,
                        'min': 0,
                        'max': 0,
                        'avg': 0
                    }
                    continue
                
                result[name] = {
                    'current': depths[-1],
                    'min': min(depths),
                    'max': max(depths),
                    'avg': sum(depths) / len(depths)
                }
            
            return result
    
    # Resource Utilization Metrics
    
    def collect_resource_metrics(self) -> None:
        """Collect current resource utilization metrics."""
        if self.config.monitor_cpu:
            self._collect_cpu_metrics()
        
        if self.config.monitor_memory:
            self._collect_memory_metrics()
        
        if self.config.monitor_disk:
            self._collect_disk_metrics()
        
        if self.config.monitor_network:
            self._collect_network_metrics()
        
        if self.config.monitor_gpu and GPU_AVAILABLE:
            self._collect_gpu_metrics()
    
    def get_cpu_utilization(self, process_id: Optional[int] = None) -> float:
        """
        Get current CPU utilization.
        
        Args:
            process_id: Optional process ID to get utilization for, or None for system
            
        Returns:
            CPU utilization as a percentage
        """
        if process_id:
            try:
                process = psutil.Process(process_id)
                return process.cpu_percent(interval=0.1)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return 0.0
        else:
            return psutil.cpu_percent(interval=0.1)
    
    def get_memory_utilization(self, process_id: Optional[int] = None) -> Dict[str, float]:
        """
        Get current memory utilization.
        
        Args:
            process_id: Optional process ID to get utilization for, or None for system
            
        Returns:
            Dictionary with memory metrics (used, available, percent)
        """
        if process_id:
            try:
                process = psutil.Process(process_id)
                memory_info = process.memory_info()
                return {
                    'rss': memory_info.rss,
                    'vms': memory_info.vms,
                    'percent': process.memory_percent()
                }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return {'rss': 0, 'vms': 0, 'percent': 0}
        else:
            memory = psutil.virtual_memory()
            return {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'percent': memory.percent
            }
    
    def get_disk_utilization(self) -> Dict[str, Dict[str, float]]:
        """
        Get current disk utilization.
        
        Returns:
            Dictionary mapping disk paths to utilization metrics
        """
        result = {}
        
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                result[partition.mountpoint] = {
                    'total': usage.total,
                    'used': usage.used,
                    'free': usage.free,
                    'percent': usage.percent
                }
            except (PermissionError, FileNotFoundError):
                pass
        
        return result
    
    def get_network_utilization(self) -> Dict[str, float]:
        """
        Get current network utilization.
        
        Returns:
            Dictionary with network metrics (bytes_sent, bytes_recv, etc.)
        """
        net_io = psutil.net_io_counters()
        
        result = {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv,
            'errin': net_io.errin,
            'errout': net_io.errout,
            'dropin': net_io.dropin,
            'dropout': net_io.dropout
        }
        
        # Calculate rates if we have previous measurements
        if self._last_net_io and self._last_net_io_time:
            current_time = time.time()
            time_diff = current_time - self._last_net_io_time
            
            if time_diff > 0:
                result['bytes_sent_per_sec'] = (net_io.bytes_sent - self._last_net_io.bytes_sent) / time_diff
                result['bytes_recv_per_sec'] = (net_io.bytes_recv - self._last_net_io.bytes_recv) / time_diff
        
        return result
    
    def get_gpu_utilization(self) -> Dict[int, Dict[str, float]]:
        """
        Get current GPU utilization.
        
        Returns:
            Dictionary mapping GPU IDs to utilization metrics
        """
        result = {}
        
        if not GPU_AVAILABLE:
            return result
        
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                result[i] = {
                    'utilization': gpu.load * 100,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'temperature': gpu.temperature
                }
        except Exception:
            pass
        
        return result
    
    def estimate_inference_metrics(self, model_name: str, input_token_count: int, output_token_count: int, 
                                  hardware_config: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Estimate inference metrics like TTFT, throughput, and memory usage based on model and hardware parameters.
        
        Uses the formula:
        TTFT ≈ T_prefill = (2 × Parameters × N_input)/(Effective FLOPS) + (Activation Memory)/(Memory Bandwidth)
        
        Args:
            model_name: Name of the model to estimate for
            input_token_count: Number of input tokens
            output_token_count: Number of output tokens
            hardware_config: Optional hardware configuration with keys:
                            'gpu_flops': GPU FLOPS in operations per second
                            'memory_bandwidth': Memory bandwidth in bytes per second
                            'hardware_efficiency': Efficiency factor (0.0-1.0) of achieved vs theoretical FLOPS
        
        Returns:
            Dictionary with estimated metrics:
                'ttft_seconds': Time to first token in seconds
                'total_time_seconds': Total inference time in seconds
                'tokens_per_second': Estimated token generation rate
                'memory_usage_bytes': Total memory usage in bytes
        """
        # Get hardware configuration or use defaults
        hw_config = hardware_config or {}
        gpu_flops = hw_config.get('gpu_flops', self.config.default_gpu_flops)  # Operations/second
        memory_bandwidth = hw_config.get('memory_bandwidth', self.config.default_memory_bandwidth)  # Bytes/second
        hardware_efficiency = hw_config.get('hardware_efficiency', 0.4)  # 40% of theoretical peak is typical
        
        # Use MemoryInstrumenter to get model memory estimates
        memory_instrumentation = MemoryInstrumenter()
        memory_estimates = memory_instrumentation.estimate_model_memory_cost(
            model_name, input_token_count, output_token_count
        )
        
        # Extract needed values
        parameter_count = memory_estimates['parameter_memory'] / 2  # Convert from bytes back to parameter count
        activation_memory = memory_estimates['activated_memory']  # In bytes
        
        # Calculate TTFT based on the formula
        # TTFT ≈ T_prefill = (2 × Parameters × N_input)/(Effective FLOPS) + (Activation Memory)/(Memory Bandwidth)
        compute_time = (2 * parameter_count * input_token_count) / gpu_flops
        memory_time = activation_memory / memory_bandwidth
        ttft_seconds = compute_time + memory_time
        
        # Calculate tokens_per_second based on model size and hardware
        # During generation, computation is roughly proportional to parameter count
        # with ~2 FLOPs per parameter per output token for decoder models
        effective_flops = gpu_flops * hardware_efficiency
        operations_per_token = 2 * parameter_count  # ~2 FLOPs per parameter per token for generation
        tokens_per_second = effective_flops / operations_per_token
        
        # Apply bandwidth limits - generation can also be memory-bound
        kv_cache_per_token = memory_estimates['kv_cache'] / (input_token_count + output_token_count)
        memory_bound_tokens_per_second = memory_bandwidth / kv_cache_per_token
        
        # Use the lower of compute-bound or memory-bound token generation rate
        tokens_per_second = min(tokens_per_second, memory_bound_tokens_per_second)
        
        # Estimate total inference time (TTFT + generation time)
        generation_time = output_token_count / tokens_per_second
        total_time_seconds = ttft_seconds + generation_time
        
        # Calculate effective tokens per second (considering the entire process)
        effective_tokens_per_second = output_token_count / total_time_seconds if total_time_seconds > 0 else 0
        
        return {
            'ttft_seconds': ttft_seconds,
            'total_time_seconds': total_time_seconds,
            'memory_usage_bytes': memory_estimates['total'],
            'parameter_memory_bytes': memory_estimates['parameter_memory'],
            'activation_memory_bytes': memory_estimates['activated_memory'],
            'kv_cache_bytes': memory_estimates['kv_cache']
        } 