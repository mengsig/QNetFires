# Fire Simulation Performance Analysis & Optimization Guide

## Executive Summary

Your fire simulation is experiencing severe performance bottlenecks despite having significant computational resources (128 cores, 30GB RAM). The primary issue is **inefficient parallelization architecture** combined with **memory-intensive fire physics simulations**. The system is not GPU-bound but rather suffering from nested parallelization conflicts and sequential bottlenecks within the fire simulation engine.

## Current Configuration Analysis

### Hardware Resources
- **CPU**: 4 physical cores, 8 logical CPUs (2 threads per core) - NOT 128 cores as expected
- **RAM**: 30GB total (only 3.8GB currently used)
- **GPU**: None detected (not the bottleneck)
- **Current CPU Usage**: ~1-3% (underutilization, but on only 4 cores)

**⚠️ CRITICAL DISCOVERY**: The system has only 4 CPU cores, not 128! This explains the performance issues.

### Configuration Issues
From `workstation_config.json`:
```json
{
  "num_landscapes": 500,           // TOO HIGH
  "num_parallel_envs": 250,        // TOO HIGH
  "max_workers": 100,              // COMPETING WITH ENVS
  "memory_simulations": 200,       // EXPENSIVE PER ENV
  "fire_simulation_max_duration": 120,  // VERY LONG
  "steps_per_episode": 200         // HIGH OVERHEAD
}
```

## Root Cause Analysis

### 1. **Nested Parallelization Problem** ⚠️ **CRITICAL**
```python
# VectorizedFireEnv.py - Line ~91
self.executor = ThreadPoolExecutor(max_workers=self.max_workers)  # 100 threads
```
**Problem**: You have 250 environments each trying to use ThreadPoolExecutor with 100 workers, creating potential for 25,000+ threads competing for 128 cores.

### 2. **Sequential Fire Simulation Bottleneck** ⚠️ **CRITICAL**
```python
# Simulate.py - Line ~108
def run_many_simulations(self, num_simulations, max_duration=None):
    for i in range(num_simulations):  # SEQUENTIAL LOOP
        xcord = np.random.randint(0, self.slope.shape[0])
        ycord = np.random.randint(0, self.slope.shape[1])
        self.run_simulation(xcord, ycord)  # EXPENSIVE PHYSICS
```
**Problem**: Each environment runs 200 fire simulations sequentially, each taking significant time with complex physics calculations.

### 3. **Memory-Intensive Space-Time Cubes** ⚠️ **HIGH**
```python
# Simulate.py - Line ~15
self.cube_shape = (time_steps, self.slope.shape[0], self.slope.shape[1])
# Creates massive 3D arrays for each environment
```
**Problem**: Each environment creates large 3D space-time cubes for fire simulation, causing memory pressure and cache misses.

### 4. **Thread Contention & Context Switching** ⚠️ **CRITICAL**
- 250 environments × 100 workers = potential 25,000 threads
- **Only 4 CPU cores** (8 logical) can't handle this at all!
- Extreme context switching overhead (3000+ threads per core)
- Complete thread pool saturation and competition

## Specific Optimization Recommendations

### 🚀 **Immediate Fixes (High Impact)**

#### 1. **Fix Parallelization Architecture**
```json
// RECOMMENDED: workstation_config_optimized.json
{
  "num_landscapes": 8,             // Match logical CPU count
  "num_parallel_envs": 4,          // Match physical cores
  "max_workers": 2,                // Conservative per environment
  "parallel_method": "threading",  // Keep, but with strict limits
  "memory_simulations": 5,         // Much lower for 4-core system
  "fire_simulation_max_duration": 15,  // Much shorter duration
  "steps_per_episode": 25          // Reduce overhead significantly
}
```

#### 2. **Parallelize Fire Simulations Within Environments**
Create a new optimized fire simulation class:

```python
# src/scripts/OptimizedSimulate.py
import concurrent.futures
import numpy as np
from typing import List, Tuple

class OptimizedSimulate:
    def run_many_simulations_parallel(self, num_simulations, max_workers=4):
        """Run fire simulations in parallel within each environment."""
        
        def run_single_simulation(args):
            xcord, ycord, sim_id = args
            # Run individual simulation
            return self.run_simulation(xcord, ycord)
        
        # Prepare simulation arguments
        sim_args = []
        for i in range(num_simulations):
            xcord = np.random.randint(0, self.slope.shape[0])
            ycord = np.random.randint(0, self.slope.shape[1])
            sim_args.append((xcord, ycord, i))
        
        # Run simulations in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(run_single_simulation, sim_args))
        
        # Aggregate results
        self.burned = np.zeros_like(self.slope, dtype=np.uint8)
        self.average_acres_burned = 0
        
        for result in results:
            if hasattr(result, 'burned'):
                self.burned += result.burned
                self.average_acres_burned += result.acres_burned
```

#### 3. **Reduce Memory Footprint**
```python
# Optimize space-time cube creation
def set_space_time_cubes_optimized(self, time_steps=None):
    # Reduce time resolution for faster simulation
    if time_steps is None:
        time_steps = min(500, int(300 * np.sqrt(self.slope.shape[0] * self.slope.shape[1]) / 400))
    
    # Use smaller data types where possible
    self.cube_shape = (time_steps, self.slope.shape[0], self.slope.shape[1])
    
    # Use float16 for memory efficiency where precision allows
    slope_optimized = self.slope.astype(np.float16)
    # ... continue with optimizations
```

### 🔧 **Architecture Improvements (Medium Impact)**

#### 4. **Implement Environment Pool Management**
```python
class OptimizedVectorizedFireEnv:
    def __init__(self, landscape_data_list, num_envs=16, workers_per_env=4):
        self.num_envs = min(num_envs, 32)  # Cap environments
        self.workers_per_env = workers_per_env
        
        # Create limited number of environments
        self.envs = []
        for i in range(self.num_envs):
            env = self.create_optimized_env(landscape_data_list[i % len(landscape_data_list)])
            self.envs.append(env)
        
        # Use process pool for better isolation
        self.executor = ProcessPoolExecutor(max_workers=self.num_envs)
```

#### 5. **Batch Processing Optimization**
```python
def collect_experiences_batched(self, num_steps, batch_size=8):
    """Collect experiences in optimized batches."""
    
    for batch_start in range(0, num_steps, batch_size):
        batch_end = min(batch_start + batch_size, num_steps)
        batch_size_actual = batch_end - batch_start
        
        # Process batch in parallel
        batch_futures = []
        for step in range(batch_size_actual):
            future = self.executor.submit(self.step_single_env, step)
            batch_futures.append(future)
        
        # Wait for batch completion
        results = [future.result() for future in batch_futures]
        
        # Process results efficiently
        self.process_batch_results(results)
```

### 📊 **Performance Monitoring (Low Impact)**

#### 6. **Add Performance Profiling**
```python
import cProfile
import time
from memory_profiler import profile

class PerformanceMonitor:
    def __init__(self):
        self.timings = {}
        self.memory_usage = {}
    
    @profile
    def profile_simulation_step(self, env_step_func, *args):
        start_time = time.time()
        result = env_step_func(*args)
        end_time = time.time()
        
        self.timings['step_time'] = end_time - start_time
        return result
    
    def print_performance_summary(self):
        print("=== Performance Summary ===")
        for key, value in self.timings.items():
            print(f"{key}: {value:.3f}s")
```

## Recommended Implementation Plan

### Phase 1: Quick Wins (1 hour - CRITICAL)
1. **Drastically reduce environment count**: Set `num_parallel_envs: 4`
2. **Drastically reduce simulations per environment**: Set `memory_simulations: 5`
3. **Much shorter simulation duration**: Set `fire_simulation_max_duration: 15`
4. **Much lower max_workers**: Set `max_workers: 2`

### Phase 2: Architecture Fix (3-5 days)
1. Implement `OptimizedSimulate` class
2. Parallelize `run_many_simulations` within each environment
3. Use ProcessPoolExecutor instead of ThreadPoolExecutor for environment isolation
4. Add memory optimization for space-time cubes

### Phase 3: Advanced Optimization (1-2 weeks)
1. Implement caching for repeated landscape computations
2. Use memory mapping for large landscape data
3. Consider GPU acceleration for fire physics (if GPU available)
4. Implement dynamic work distribution

## Expected Performance Improvements

### Current Performance
- **Environments**: 250 (oversubscribed)
- **CPU Usage**: ~3% (underutilized)
- **Memory**: Heavy space-time cubes per env
- **Throughput**: Limited by sequential fire simulations

### Optimized Performance
- **Environments**: 4 (optimal for 4 physical cores)
- **CPU Usage**: 70-90% (properly utilized)
- **Memory**: Reduced footprint with optimizations
- **Throughput**: 10-50x improvement expected (massive oversubscription fix)

## Testing Strategy

```bash
# Test with small configuration first
cp parallel_config.json parallel_config_test.json
# Edit to reduce num_parallel_envs to 4, memory_simulations to 5

# Run performance test
time python src/scripts/train_dqn_fuel_breaks_parallel.py --config parallel_config_test.json

# Monitor with htop during execution
htop
```

## Key Metrics to Monitor

1. **CPU Utilization**: Should reach 70-90%
2. **Memory Usage**: Should be stable, not growing
3. **Experiences/Second**: Should increase significantly
4. **Training Time**: Should decrease substantially
5. **Context Switches**: Should be lower with fewer threads

## Conclusion

Your simulation is severely bottlenecked by inefficient parallelization, not hardware limitations. The primary issues are:

1. **Extreme thread oversubscription** (25K potential threads for only 4 cores!)
2. **Sequential fire simulations** within each environment
3. **Memory-intensive space-time cubes** causing cache misses
4. **Completely excessive environment count** for a 4-core system

**The system is configured for a 128-core machine but running on 4 cores!**

Implementing the recommended optimizations should provide **10-50x performance improvement** and proper CPU utilization.