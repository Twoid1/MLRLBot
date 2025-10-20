"""
GPU Optimization Utilities
- Auto-detect optimal settings
- Memory management
- Batch size optimization
- Performance benchmarking
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
import logging
import psutil
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class GPUOptimizer:
    """
    Automatically optimize GPU/CPU settings for maximum performance
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_available = torch.cuda.is_available()
        self.system_info = self._detect_system()
    
    def _detect_system(self) -> Dict:
        """Detect system capabilities"""
        info = {
            'cpu_count': psutil.cpu_count(logical=False),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'ram_gb': psutil.virtual_memory().total / (1024**3),
            'gpu_available': self.gpu_available
        }
        
        if self.gpu_available:
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info['cuda_version'] = torch.version.cuda
            info['gpu_count'] = torch.cuda.device_count()
        
        return info
    
    def get_optimal_config(self) -> Dict:
        """
        Get optimal configuration based on system capabilities
        
        Returns:
            Optimized config dict
        """
        config = {}
        
        # CPU workers
        cpu_cores = self.system_info['cpu_count']
        config['num_workers'] = max(4, min(cpu_cores - 2, 12))  # Leave 2 cores free
        
        # Batch size based on available memory
        if self.gpu_available:
            gpu_mem_gb = self.system_info['gpu_memory_gb']
            
            if gpu_mem_gb >= 24:  # High-end GPU (RTX 4090, A100, etc.)
                config['batch_size'] = 512
                config['rl_batch_size'] = 512
                config['hidden_dims'] = [512, 512, 256, 128]
            elif gpu_mem_gb >= 12:  # Mid-range GPU (RTX 3080, 4070 Ti, etc.)
                config['batch_size'] = 256
                config['rl_batch_size'] = 256
                config['hidden_dims'] = [256, 256, 128]
            elif gpu_mem_gb >= 8:  # Entry GPU (RTX 3060, 4060, etc.)
                config['batch_size'] = 128
                config['rl_batch_size'] = 128
                config['hidden_dims'] = [256, 128]
            else:  # Low-end GPU
                config['batch_size'] = 64
                config['rl_batch_size'] = 64
                config['hidden_dims'] = [128, 128]
            
            # XGBoost GPU parameters
            config['ml_gpu_params'] = {
                'tree_method': 'gpu_hist',
                'gpu_id': 0,
                'predictor': 'gpu_predictor',
                'max_bin': 256  # Optimize for GPU
            }
            
            config['use_gpu'] = True
            
        else:  # CPU-only
            ram_gb = self.system_info['ram_gb']
            
            if ram_gb >= 32:
                config['batch_size'] = 128
                config['rl_batch_size'] = 128
            else:
                config['batch_size'] = 64
                config['rl_batch_size'] = 64
            
            config['hidden_dims'] = [256, 128]
            config['use_gpu'] = False
            config['ml_gpu_params'] = {}
        
        # Training optimization
        config['rl_update_frequency'] = 4 if self.gpu_available else 8
        config['preload_data'] = self.system_info['ram_gb'] >= 16
        config['cache_features'] = True

        config['precompute_observations'] = True
        
        return config
    
    def optimize_pytorch(self):
        """Optimize PyTorch settings"""
        if self.gpu_available:
            # Enable TF32 on Ampere GPUs for faster training
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Benchmark mode for faster conv operations
            torch.backends.cudnn.benchmark = True
            
            # Set memory allocator
            if hasattr(torch.cuda, 'memory'):
                torch.cuda.empty_cache()
            
            logger.info("PyTorch GPU optimizations enabled")
        else:
            # CPU optimizations
            torch.set_num_threads(self.system_info['cpu_count_logical'])
            logger.info(f"PyTorch using {self.system_info['cpu_count_logical']} CPU threads")
    
    def print_system_info(self):
        """Print detailed system information"""
        print("\n" + "="*80)
        print("SYSTEM CONFIGURATION")
        print("="*80)
        
        print(f"\nCPU:")
        print(f"  Physical Cores: {self.system_info['cpu_count']}")
        print(f"  Logical Cores:  {self.system_info['cpu_count_logical']}")
        print(f"  RAM:            {self.system_info['ram_gb']:.1f} GB")
        
        if self.gpu_available:
            print(f"\nGPU:")
            print(f"  Device:         {self.system_info['gpu_name']}")
            print(f"  Memory:         {self.system_info['gpu_memory_gb']:.1f} GB")
            print(f"  CUDA Version:   {self.system_info['cuda_version']}")
            print(f"  GPU Count:      {self.system_info['gpu_count']}")
        else:
            print(f"\nGPU: Not available (CPU-only mode)")
        
        print("\n" + "="*80)
    
    def benchmark_operations(self) -> Dict:
        """
        Benchmark key operations to estimate training time
        
        Returns:
            Benchmark results
        """
        print("\nRunning performance benchmarks...")
        results = {}
        
        # Benchmark matrix multiplication
        size = 1000
        iterations = 100
        
        if self.gpu_available:
            # GPU benchmark
            x = torch.randn(size, size).to(self.device)
            torch.cuda.synchronize()
            
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            for _ in range(iterations):
                y = torch.mm(x, x)
            end.record()
            
            torch.cuda.synchronize()
            gpu_time = start.elapsed_time(end) / iterations
            results['gpu_matmul_ms'] = gpu_time
            
            # GPU memory bandwidth
            results['gpu_memory_gb'] = self.system_info['gpu_memory_gb']
            
        # CPU benchmark
        import time
        x = torch.randn(size, size)
        start = time.time()
        for _ in range(iterations):
            y = torch.mm(x, x)
        cpu_time = (time.time() - start) / iterations * 1000
        results['cpu_matmul_ms'] = cpu_time
        
        if self.gpu_available:
            results['speedup'] = cpu_time / gpu_time
            print(f"  GPU: {gpu_time:.2f}ms per operation")
            print(f"  CPU: {cpu_time:.2f}ms per operation")
            print(f"  GPU Speedup: {results['speedup']:.1f}x")
        else:
            print(f"  CPU: {cpu_time:.2f}ms per operation")
        
        return results
    
    def estimate_training_time(self, config: Dict) -> Dict:
        """
        Estimate total training time based on configuration
        
        Args:
            config: Training configuration
            
        Returns:
            Time estimates in hours
        """
        estimates = {}
        
        # Base estimates (from empirical testing)
        if self.gpu_available:
            gpu_mem = self.system_info['gpu_memory_gb']
            
            # Data loading (parallelized)
            estimates['data_loading_min'] = 2
            
            # Feature calculation (parallelized)
            n_assets = len(config.get('assets', []))
            estimates['feature_calc_min'] = n_assets * 0.5  # 30s per asset parallel
            
            # ML training (GPU accelerated)
            estimates['ml_training_min'] = 15 if gpu_mem >= 12 else 30
            
            # RL training (GPU accelerated)
            episodes = config.get('rl_episodes', 100)
            # Approximately 1-2 seconds per episode on GPU
            estimates['rl_training_min'] = (episodes * 1.5) / 60
            
        else:  # CPU only
            estimates['data_loading_min'] = 5
            estimates['feature_calc_min'] = len(config.get('assets', [])) * 2
            estimates['ml_training_min'] = 60
            episodes = config.get('rl_episodes', 100)
            estimates['rl_training_min'] = (episodes * 10) / 60
        
        # Total
        estimates['total_hours'] = sum(estimates.values()) / 60
        
        return estimates
    
    def save_optimal_config(self, output_path: str = 'optimal_config.json'):
        """Save optimal configuration to file"""
        config = self.get_optimal_config()
        config['system_info'] = self.system_info
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Optimal config saved to {output_path}")
        return config


class MemoryMonitor:
    """Monitor GPU/CPU memory usage during training"""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.peak_memory = 0
    
    def get_current_usage(self) -> Dict:
        """Get current memory usage"""
        usage = {
            'ram_gb': psutil.virtual_memory().used / (1024**3),
            'ram_percent': psutil.virtual_memory().percent
        }
        
        if self.gpu_available:
            usage['gpu_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
            usage['gpu_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
            usage['gpu_max_allocated_gb'] = torch.cuda.max_memory_allocated() / (1024**3)
        
        return usage
    
    def print_usage(self):
        """Print current memory usage"""
        usage = self.get_current_usage()
        
        print(f"\nMemory Usage:")
        print(f"  RAM: {usage['ram_gb']:.2f} GB ({usage['ram_percent']:.1f}%)")
        
        if self.gpu_available:
            print(f"  GPU Allocated: {usage['gpu_allocated_gb']:.2f} GB")
            print(f"  GPU Reserved:  {usage['gpu_reserved_gb']:.2f} GB")
            print(f"  GPU Peak:      {usage['gpu_max_allocated_gb']:.2f} GB")
    
    def clear_cache(self):
        """Clear GPU cache"""
        if self.gpu_available:
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")


def setup_optimal_training(config_path: Optional[str] = None) -> Dict:
    """
    Setup optimal training configuration
    
    Usage:
        config = setup_optimal_training()
        trainer = OptimizedSystemTrainer(config)
    
    Returns:
        Optimal configuration dict
    """
    optimizer = GPUOptimizer()
    
    # Print system info
    optimizer.print_system_info()
    
    # Optimize PyTorch
    optimizer.optimize_pytorch()
    
    # Get optimal config
    config = optimizer.get_optimal_config()
    
    # Load base config if provided
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            base_config = json.load(f)
            config.update(base_config)
    
    # Add default training parameters
    config.setdefault('assets', ['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'ADA_USDT', 'DOT_USDT'])
    config.setdefault('timeframes', ['1h', '4h', '1d'])
    config.setdefault('rl_episodes', 100)
    
    # Benchmark performance
    benchmarks = optimizer.benchmark_operations()
    
    # Estimate training time
    estimates = optimizer.estimate_training_time(config)
    
    print("\n" + "="*80)
    print("TRAINING TIME ESTIMATES")
    print("="*80)
    print(f"Data Loading:      {estimates['data_loading_min']:.1f} minutes")
    print(f"Feature Calc:      {estimates['feature_calc_min']:.1f} minutes")
    print(f"ML Training:       {estimates['ml_training_min']:.1f} minutes")
    print(f"RL Training:       {estimates['rl_training_min']:.1f} minutes")
    print("-"*80)
    print(f"TOTAL ESTIMATED:   {estimates['total_hours']:.2f} hours")
    print("="*80)
    
    # Save optimal config
    optimizer.save_optimal_config('config/optimal_training_config.json')
    print(f"\nOptimal config saved to: config/optimal_training_config.json")
    
    return config


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='GPU Optimization Utilities')
    parser.add_argument('--info', action='store_true', help='Show system info')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmarks')
    parser.add_argument('--config', action='store_true', help='Generate optimal config')
    parser.add_argument('--estimate', action='store_true', help='Estimate training time')
    
    args = parser.parse_args()
    
    optimizer = GPUOptimizer()
    
    if args.info or not any([args.benchmark, args.config, args.estimate]):
        optimizer.print_system_info()
    
    if args.benchmark:
        optimizer.benchmark_operations()
    
    if args.config:
        config = optimizer.get_optimal_config()
        print("\nOptimal Configuration:")
        print(json.dumps(config, indent=2))
        optimizer.save_optimal_config()
    
    if args.estimate:
        config = optimizer.get_optimal_config()
        config['assets'] = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'ADA_USDT', 'DOT_USDT']
        config['rl_episodes'] = 100
        
        estimates = optimizer.estimate_training_time(config)
        print("\nTraining Time Estimates:")
        for key, value in estimates.items():
            if 'min' in key:
                print(f"  {key}: {value:.1f} minutes")
            else:
                print(f"  {key}: {value:.2f} hours")