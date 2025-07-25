# QAgent Training Troubleshooting Guide

## 🚨 Common Errors and Solutions

### 1. **EOFError / Multiprocessing Crashes**

**Error:**
```
EOFError
Exception ignored in: <function AsyncVectorEnv.__del__ at 0x...>
```

**Cause:** One of the environment processes crashed, usually due to fire simulation errors.

**Solutions (try in order):**

#### ✅ **Option 1: Use Robust Training Script**
```bash
python3 train_robust.py
```
This automatically handles crashes and restarts training.

#### ✅ **Option 2: Reduce Environment Complexity**
Edit `src/Train.py`:
```python
N_ENVS = 8              # Reduce from 16
SIMS = 5                # Reduce from 10  
USE_ENHANCED_MODEL = False  # Use basic model
```

#### ✅ **Option 3: Check Fire Simulation Dependencies**
```bash
# Make sure pyretechnics is properly installed
pip install -e pyretechnics/.
```

---

### 2. **CUDA Out of Memory**

**Error:**
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate X MiB.
```

**Solutions (try in order):**

#### ✅ **Option 1: Use Memory-Efficient Script**
```bash
python3 train_memory_efficient.py
```

#### ✅ **Option 2: Manual Memory Optimization**
Edit `src/Train.py`:
```python
BATCH_SIZE = 16         # Reduce from 32
N_ENVS = 8             # Reduce from 16
MEMORY_EFFICIENT = True # Enable optimizations
```

#### ✅ **Option 3: Environment Variable**
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 src/Train.py
```

---

### 3. **Model Initialization Error**

**Error:**
```
AttributeError: 'NoneType' object has no attribute 'fill_'
```

**Solution:** ✅ **Fixed in latest version**
This was a bug in weight initialization that has been resolved.

---

### 4. **Mean Reward = NaN**

**Symptoms:**
```
mean_reward=nan
```

**Cause:** No episodes are completing successfully.

**Solutions:**
1. **Check environment stability** - use robust training script
2. **Reduce episode length** - decrease `BUDGET` parameter
3. **Check reward calculation** - environments may be crashing

---

### 5. **Training Very Slow**

**Solutions (try in order):**

#### ✅ **Option 1: Reduce Fire Simulations**
```python
SIMS = 5  # Reduce from 25
```

#### ✅ **Option 2: Use Basic Model**
```python
USE_ENHANCED_MODEL = False
```

#### ✅ **Option 3: Fewer Environments**
```python
N_ENVS = 8  # Reduce from 16
```

---

### 6. **No Rasters Found**

**Symptoms:**
```
Warning: Could not load raster X: [Errno 2] No such file or directory
```

**Solution:** ✅ **Automatic Fallback**
The system automatically creates dummy rasters for testing. Training will continue.

To use real rasters, ensure your raster files are in the correct directory structure:
```
cropped_raster/
├── slp/
├── asp/
├── dem/
├── cc/
├── cbd/
├── cbh/
├── ch/
├── fbfm/
└── fireline/
```

---

## 🛠️ **Diagnostic Commands**

### Check System Status
```bash
# Test model architectures
python3 test_model_fix.py

# Test overall structure  
python3 simple_test.py

# Check memory configuration
python3 memory_config.py
```

### Check GPU Memory
```bash
nvidia-smi  # Check GPU usage
```

### Check Multiprocessing
```bash
ps aux | grep python  # Check for hanging processes
```

---

## 🎯 **Recommended Training Configurations**

### **For 2-4GB GPU (Most Common)**
```python
BATCH_SIZE = 16
N_ENVS = 8
SIMS = 5
USE_ENHANCED_MODEL = True
MEMORY_EFFICIENT = True
```

### **For < 2GB GPU (Limited)**
```python
BATCH_SIZE = 8
N_ENVS = 4
SIMS = 3
USE_ENHANCED_MODEL = False
```

### **For > 6GB GPU (High-End)**
```python
BATCH_SIZE = 64
N_ENVS = 32
SIMS = 25
USE_ENHANCED_MODEL = True
MEMORY_EFFICIENT = False
```

---

## 🚀 **Quick Fix Commands**

### **For Most Issues:**
```bash
python3 train_robust.py
```

### **For Memory Issues:**
```bash
python3 train_memory_efficient.py
```

### **For Emergency Fallback:**
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 src/Train.py
```

---

## 📞 **Still Having Issues?**

1. **Check the error type** in the list above
2. **Try the robust training script** first
3. **Reduce complexity** (fewer environments, simulations)
4. **Use basic model** if enhanced model causes issues
5. **Check dependencies** are properly installed

The system is designed to be resilient and should work even with limited resources or missing dependencies through automatic fallbacks.