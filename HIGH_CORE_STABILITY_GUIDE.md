# High-Core Workstation Stability Guide

## Overview
This guide provides two stable training solutions optimized for your 128-core workstation with 32+ parallel environments.

## 🚀 Quick Start Options

### Option 1: New Stable Script (Recommended)
```bash
python3 src/Train_stable.py
```
**Features:**
- ✅ Custom `StableProcessPoolEnv` using `ProcessPoolExecutor`
- ✅ Built-in timeout handling and process management
- ✅ Optimized for 32 environments with 16 worker processes
- ✅ Enhanced error recovery and environment recreation
- ✅ No `AsyncVectorEnv` hanging issues

### Option 2: Patched Original Script
```bash
# Apply stability patches to existing src/Train.py
python3 patch_train_stable.py

# Then run the patched version
python3 src/Train.py
```
**Features:**
- ✅ All stability fixes applied to your existing `src/Train.py`
- ✅ Maintains your existing code structure
- ✅ Backup created automatically
- ✅ 32 environments with enhanced error handling

## 🔧 Key Stability Improvements

### 1. Process Pool Architecture (`Train_stable.py`)
```python
class StableProcessPoolEnv:
    def __init__(self, env_fns, max_workers=None, timeout=15.0):
        self.executor = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp.get_context('spawn')
        )
```
**Benefits:**
- No pipe corruption issues
- Built-in timeout handling
- Automatic process cleanup
- Better resource management

### 2. Environment Reset Fix
**Problem:** Environments stuck in "done" state between episodes
**Solution:** Added mandatory reset between episodes
```python
# Reset environments for next episode (CRITICAL FIX)
obs = vec_env.reset()
```

### 3. Optimized Parameters for 128-Core System
```python
N_ENVS = 32              # Utilize high-core count
STEPS_PER_EP = 25        # Allow proper episode completion
BUFFER_CAP = 200_000     # Large buffer for high throughput
BATCH_SIZE = 64          # Efficient batch processing
SIMS = 1                 # Minimal simulations for stability
```

### 4. Enhanced Error Handling
- **Timeout Reduction**: 8-second timeouts for faster error detection
- **Process Cleanup**: Using `psutil` for aggressive cleanup
- **Graceful Degradation**: Fallback mechanisms for failed environments
- **Automatic Recovery**: Environment recreation on critical failures

### 5. Action Data Type Fix
```python
action = np.zeros(HxW, dtype=np.int8)  # Correct data type for FuelBreakEnv
```

## 📊 Performance Expectations

### High-Core Utilization
- **32 Parallel Environments**: Full utilization of your 128-core system
- **16 Worker Processes**: Optimal balance between parallelism and stability
- **200K Buffer Capacity**: Handles high-throughput training data

### Stability Metrics
- **No Hanging**: Robust timeout and recovery mechanisms
- **Proper Episodes**: 25 steps per episode allowing natural completion
- **Error Recovery**: Automatic environment recreation on failures
- **Memory Efficiency**: Optimized for long training runs

## 🛠️ Troubleshooting

### If Training Still Hangs
1. **Reduce N_ENVS**: Try 16 or 24 environments
2. **Check Memory**: Monitor RAM usage (32 envs can use significant memory)
3. **Verify Rasters**: Ensure all 500 rasters load correctly

### Memory Optimization
```python
# For systems with limited RAM
N_ENVS = 16              # Reduce environments
BATCH_SIZE = 32          # Smaller batches
BUFFER_CAP = 100_000     # Smaller buffer
```

### Process Monitoring
```bash
# Monitor training processes
htop
# or
ps aux | grep python
```

## 🔍 Comparison: AsyncVectorEnv vs StableProcessPoolEnv

| Feature | AsyncVectorEnv | StableProcessPoolEnv |
|---------|----------------|---------------------|
| **Stability** | ❌ Prone to hanging | ✅ Robust with timeouts |
| **Error Recovery** | ❌ Manual cleanup needed | ✅ Automatic recovery |
| **Process Management** | ❌ Complex pipe handling | ✅ Built-in management |
| **High-Core Scaling** | ❌ Issues with 32+ envs | ✅ Designed for 32+ envs |
| **Memory Efficiency** | ❌ Can leak processes | ✅ Clean shutdown |

## 🎯 Recommended Configuration for 128-Core System

### Conservative (Most Stable)
```python
N_ENVS = 16
BATCH_SIZE = 32
SIMS = 1
```

### Balanced (Recommended)
```python
N_ENVS = 32
BATCH_SIZE = 64
SIMS = 1
```

### Aggressive (Maximum Performance)
```python
N_ENVS = 48
BATCH_SIZE = 96
SIMS = 1
```

## 📈 Expected Training Output

### Successful Training Session
```
🚀 Using device cuda on 128-core system
🎯 Training with 32 environments on 128 cores
📂 Loading rasters...
✅ Loaded 500 rasters
🚀 Creating StableProcessPoolEnv with 32 environments and 16 workers
🔧 Initializing 32 environments...
Environment 0: ✅
Environment 1: ✅
...
✅ Successfully initialized 32/32 environments

[Episode 0] Reward: -0.234 Loss: 12.45 Burned: 423.1 Eps: 1.000 Completed: 32/32
🎯 Environment 0 completed: Burned=345.2, Step=23
🎯 Environment 1 completed: Burned=456.7, Step=25
...
🔄 Resetting 32 environments...
[Episode 1] Reward: -0.198 Loss: 11.23 Burned: 398.7 Eps: 0.998 Completed: 32/32
```

## 🚨 Warning Signs to Watch For

### Bad Signs (Indicates Instability)
- Episodes completing in 1 step with 0 fuel breaks
- Constant "Environment step timed out" messages
- Memory usage continuously increasing
- No environment completion messages

### Good Signs (Indicates Stability)
- Episodes completing in 20-25 steps
- Fuel breaks being placed (250 per episode)
- Varied burned area values
- Regular reset messages between episodes

## 🔧 Advanced Tuning

### For Even Higher Core Counts (64+ environments)
```python
# Modify StableProcessPoolEnv
max_workers = min(N_ENVS, 32)  # Increase worker limit
timeout = 20.0                 # Longer timeout for complex envs
```

### Memory-Constrained Systems
```python
# Reduce memory usage
GRADIENT_ACCUMULATION_STEPS = 4  # Simulate larger batches
torch.cuda.empty_cache()         # Regular GPU cleanup
```

## 📝 Summary

The stability improvements address the core issues with high-core training:

1. **Process Management**: `StableProcessPoolEnv` eliminates pipe corruption
2. **Episode Completion**: Proper reset between episodes prevents stuck states  
3. **Error Recovery**: Robust timeout and cleanup mechanisms
4. **Resource Optimization**: Tuned parameters for 128-core systems
5. **Data Type Fixes**: Correct action types prevent environment failures

Choose `src/Train_stable.py` for maximum stability or patch your existing `src/Train.py` to maintain your current workflow.