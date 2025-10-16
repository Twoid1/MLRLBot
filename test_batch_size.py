"""
Simple batch size test without full RL dependencies
Tests pure PyTorch performance with different batch sizes
"""

import time
import torch
import torch.nn as nn
import numpy as np

print("="*80)
print("BATCH SIZE & GPU UTILIZATION TEST")
print("="*80)

# Check GPU availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"\n GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    device = torch.device("cpu")
    print(f"\n  GPU not available, running on CPU")

# Create a DQN-like network
class TestNetwork(nn.Module):
    def __init__(self, state_dim=300, action_dim=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

# Test configurations
configs = [
    {'batch': 64, 'name': 'Small (64)'},
    {'batch': 128, 'name': 'Medium-Small (128)'},
    {'batch': 256, 'name': 'Medium (256)'},
    {'batch': 512, 'name': 'Large (512)'},
    {'batch': 1024, 'name': 'XL (1024)'},
    {'batch': 2048, 'name': 'XXL (2048)'},
]

results = []

print("\nTesting different batch sizes...")
print("Each test: 200 training iterations (forward + backward pass)\n")

for config in configs:
    batch_size = config['batch']
    
    print(f"{'='*80}")
    print(f"Testing: {config['name']}")
    print(f"{'='*80}")
    
    try:
        # Create model and optimizer
        model = TestNetwork().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        
        # Warm up
        dummy_input = torch.randn(batch_size, 300).to(device)
        for _ in range(10):
            output = model(dummy_input)
            loss = output.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Actual test
        iterations = 200
        start_time = time.time()
        
        for i in range(iterations):
            # Generate random batch
            states = torch.randn(batch_size, 300).to(device)
            
            # Forward pass
            q_values = model(states)
            
            # Compute loss
            target = torch.randn_like(q_values)
            loss = nn.MSELoss()(q_values, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Print progress
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                its_per_sec = (i + 1) / elapsed
                print(f"  Iteration {i+1:3d}/{iterations} - {its_per_sec:.1f} it/s")
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        total_time = time.time() - start_time
        iterations_per_sec = iterations / total_time
        time_per_iteration = total_time / iterations * 1000  # ms
        samples_per_sec = batch_size * iterations_per_sec
        
        # Get GPU memory if available
        gpu_mem = 0
        if device.type == 'cuda':
            gpu_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
            torch.cuda.reset_peak_memory_stats()
        
        result = {
            'name': config['name'],
            'batch_size': batch_size,
            'total_time': total_time,
            'iterations_per_sec': iterations_per_sec,
            'time_per_iteration': time_per_iteration,
            'samples_per_sec': samples_per_sec,
            'gpu_memory_mb': gpu_mem,
            'success': True
        }
        results.append(result)
        
        print(f"\n  Results:")
        print(f"    Total time: {total_time:.2f}s")
        print(f"    Speed: {iterations_per_sec:.2f} iterations/sec")
        print(f"    Time per iteration: {time_per_iteration:.2f}ms")
        print(f"    Throughput: {samples_per_sec:,.0f} samples/sec")
        if device.type == 'cuda':
            print(f"    Peak GPU memory: {gpu_mem:.0f} MB")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"  ❌ OUT OF MEMORY - Batch size {batch_size} too large")
            result = {
                'name': config['name'],
                'batch_size': batch_size,
                'success': False,
                'error': 'OOM'
            }
            results.append(result)
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        else:
            raise

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

# Print results table
print(f"\n{'Batch Size':<15} {'Speed (it/s)':<15} {'Time/It (ms)':<15} {'Throughput':<20} {'GPU Mem (MB)':<15}")
print("-"*80)

successful_results = [r for r in results if r['success']]

if not successful_results:
    print("No successful tests!")
else:
    # Find baseline
    baseline = successful_results[0]
    
    for r in successful_results:
        speedup = r['iterations_per_sec'] / baseline['iterations_per_sec']
        throughput_str = f"{r['samples_per_sec']:,.0f} samples/s"
        
        print(f"{r['name']:<15} "
              f"{r['iterations_per_sec']:<15.2f} "
              f"{r['time_per_iteration']:<15.2f} "
              f"{throughput_str:<20} "
              f"{r['gpu_memory_mb']:<15.0f} "
              f"({speedup:.2f}x)")
    
    # Find best
    best = max(successful_results, key=lambda x: x['samples_per_sec'])
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print(f"\n Best batch size: {best['batch_size']}")
    print(f"  Throughput: {best['samples_per_sec']:,.0f} samples/second")
    print(f"  GPU Memory: {best['gpu_memory_mb']:.0f} MB")
    
    # Recommend for RL training
    if best['batch_size'] <= 256:
        print(f"\n  For RL training, use:")
        print(f"    'rl_batch_size': {best['batch_size']}")
    elif best['batch_size'] == 512:
        print(f"\n  For RL training, use:")
        print(f"    'rl_batch_size': 512")
        print(f"    (Optimal GPU utilization)")
    else:
        print(f"\n  For RL training, consider:")
        print(f"    'rl_batch_size': 512")
        print(f"    (Balance between speed and memory)")
    
    print(f"\n  Update frequency:")
    print(f"    'rl_update_every': 4")
    print(f"    (Update every 4 steps)")

# Show failed tests
failed = [r for r in results if not r['success']]
if failed:
    print(f"\n  Failed tests:")
    for r in failed:
        print(f"  {r['name']}: {r['error']}")

print("\n" + "="*80)

# Show current GPU status
if device.type == 'cuda':
    print(f"\nFinal GPU Status:")
    print(f"  Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.0f} MB")
    print(f"  Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.0f} MB")
    print(f"  Max memory used: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")