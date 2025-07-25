# AsyncVectorEnv EOFError Solution

## 🚨 The Problem You're Experiencing

After 114 epochs, you're getting:
```
async_vector_env.py: line 546
vector_env.py: line 205  
async_vector_env.py: line 461
async_vector_env.py: line 32: result, success = pipe.recv()
line 250 in recv: buf = self._recv_bytes()
...
raise EOFError
```

## 🔍 Why This Happens

### **Root Cause: Multiprocessing Pipe Corruption**
- `AsyncVectorEnv` uses multiprocessing pipes for communication
- After many episodes (~100+), pipes can become corrupted
- Worker processes crash or hang, breaking the communication channel
- The main process waits forever for a response that never comes

### **Why After 114 Epochs?**
- **Memory accumulation**: Long-running processes accumulate memory issues
- **Resource exhaustion**: File descriptors, memory, CPU resources get depleted
- **Fire simulation complexity**: `pyretechnics` is computationally intensive
- **Process lifecycle**: Multiprocessing workers weren't designed for infinite loops

## 🛠️ Solutions Applied

### **1. Immediate EOFError Detection**
```python
# NEW: Immediately detect and handle pipe corruption
if isinstance(e, (EOFError, BrokenPipeError, ConnectionResetError)):
    print("🔄 Pipe corruption detected - immediately recreating all environments")
    break  # Skip retries, go straight to recreation
```

### **2. Robust Environment Recreation**
- **Aggressive process cleanup** using `psutil`
- **Fresh raster selection** for new environments
- **Complexity reduction** after repeated failures
- **Fallback to minimal setup** if recreation fails

### **3. Proactive Environment Health Checks**
- **Every 50 episodes**: Check for failure patterns
- **Every 100 episodes**: Proactively recreate environments before corruption
- **Failure tracking**: Monitor AsyncVectorEnv stability

### **4. Adaptive Complexity Reduction**
```python
# After 3 failures, reduce complexity
if main._async_failures >= 3:
    effective_n_envs = max(4, N_ENVS // 2)  # Fewer environments
    effective_sims = 1  # Simpler simulations
```

## 🎯 What You Should See Now

### **Normal Operation:**
```
✅ No AsyncVectorEnv failures detected
🔍 Environment health check at episode 150...
🔄 Proactive environment recreation at episode 200 (prevents pipe corruption)
✅ Proactively recreated 16 environments
```

### **When Issues Occur:**
```
🚨 Environment error (retry 1/3): EOFError: [Errno 32] Broken pipe
🔄 Pipe corruption detected - immediately recreating all environments
🔄 Recreating environments (failure #1)
✅ Successfully recreated all 16 environments
```

### **If Problems Persist:**
```
⚠️  5 AsyncVectorEnv failures detected so far
💡 Consider switching to train_sync.py for better stability
🔧 Reducing to 8 environments with 1 simulations
```

## 🚀 Recommended Training Strategies

### **Option 1: Continue with Enhanced AsyncVectorEnv (Current)**
- **Pros**: Maximum training speed, automatic recovery
- **Cons**: Occasional interruptions for environment recreation
- **Best for**: When you want maximum performance and can tolerate brief interruptions

### **Option 2: Switch to Synchronous Training**
```bash
python3 train_sync.py
```
- **Pros**: No multiprocessing issues, completely stable
- **Cons**: Slower training (no parallelization)
- **Best for**: When you want guaranteed stability and consistent progress

### **Option 3: Reduce Complexity**
Modify `src/Train.py`:
```python
N_ENVS = 8      # Reduce from 16
SIMS = 1        # Reduce from 2  
STEPS_PER_EP = 1  # Reduce from 2
```
- **Pros**: More stable AsyncVectorEnv, fewer resource issues
- **Cons**: Slower learning due to less parallelization
- **Best for**: Balance between speed and stability

## 📊 Monitoring Your Training

### **Healthy Signs:**
```
🔍 Environment health check at episode 150...
✅ No AsyncVectorEnv failures detected
[env 0] 🎯 Episode completed: R=2.847 L=20 Burned=86.2 Reduction=52.3% Type=REAL
```

### **Warning Signs:**
```
⚠️  3 AsyncVectorEnv failures detected so far
🔧 Reducing to 8 environments with 1 simulations
💡 Consider switching to train_sync.py for better stability
```

### **Critical Signs (Time to Switch):**
```
❌ Environment recreation failed: ...
💡 Recommendation: Stop training and use train_sync.py instead
🛑 Training cannot continue. Please use train_sync.py
```

## 🔧 Quick Fixes You Can Try

### **1. Immediate Solution**
Stop current training and restart with:
```bash
python3 train_sync.py  # Most stable option
```

### **2. Reduce Resource Usage**
In `src/Train.py`, change:
```python
N_ENVS = 8       # Reduce parallel environments
SIMS = 1         # Simplify fire simulation
```

### **3. Use Robust Async Training**
```bash
python3 train_robust_async.py  # Uses ProcessPoolExecutor instead
```

## 🎯 Bottom Line

The EOFError after 114 epochs is a **known limitation of AsyncVectorEnv** with complex simulations. The enhanced error handling will:

1. **Detect pipe corruption immediately**
2. **Recreate environments automatically** 
3. **Reduce complexity if needed**
4. **Prevent future corruption with proactive recreation**

However, if you want **guaranteed stability** without any interruptions, **`train_sync.py` is the recommended solution** for long training runs.

### **My Recommendation:**
```bash
# For maximum stability (recommended)
python3 train_sync.py

# Or continue with current enhanced async (will auto-recover)
python3 src/Train.py
```

The choice depends on whether you prefer **maximum speed with occasional interruptions** vs **consistent speed with zero interruptions**.