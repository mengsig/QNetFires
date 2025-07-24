# QAgent Enhanced Training - Quick Start Guide

## 🚀 What's New

Your QAgent repository has been enhanced with state-of-the-art deep learning techniques for wildfire fuel-break placement:

### ✅ **4 Key Improvements Delivered:**

1. **Repository Correctness** - Fixed bugs and improved code organization
2. **Enhanced Agent Architecture** - 3 model options with attention mechanisms and residual connections  
3. **Full DQN Implementation** - Prioritized replay, double DQN, target networks
4. **500-Raster Management** - Efficiently cycles through all your rasters

## 🏃‍♂️ Quick Start

### 1. Install Dependencies
```bash
# Create virtual environment (recommended)
python3 -m venv qnetfires
source qnetfires/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Test the System
```bash
# Quick structure test (no dependencies needed)
python3 simple_test.py

# Full functionality test (requires PyTorch)
python3 test_training.py
```

### 3. Start Training
```bash
# Start enhanced training
python3 src/Train.py
```

## 🎛️ Configuration Options

Edit `src/Train.py` to customize:

```python
# Model Architecture (choose one)
USE_ENHANCED_MODEL = True    # Best performance
USE_DUELING = False         # Alternative architecture
# USE_ENHANCED_MODEL = False  # Original basic model

# Advanced Features
USE_PRIORITIZED_REPLAY = True  # Better sample efficiency
USE_LR_SCHEDULER = True       # Improved convergence

# Data Management
MAX_RASTERS = 500            # Use all your rasters
RASTER_ROOT = "cropped_raster"  # Path to your raster data
```

## 📊 Model Comparison

| Model | Best For | Performance | Memory |
|-------|----------|-------------|---------|
| **EnhancedQNet** | Production use | Excellent | High |
| **DuelingQNet** | Balanced option | Very Good | Medium |
| **QNet (Original)** | Quick testing | Good | Low |

## 🔧 Troubleshooting

### Common Issues:
- **CUDA Out of Memory**: Use memory-efficient training (see below)
- **No Rasters Found**: System creates dummy data automatically
- **Slow Training**: Try `USE_ENHANCED_MODEL = False`

### 🧠 Memory Optimization (Important!)

If you get `CUDA out of memory` errors:

#### Option 1: Use Memory-Efficient Script
```bash
python3 train_memory_efficient.py
```

#### Option 2: Manual Configuration
Edit `src/Train.py` and change:
```python
BATCH_SIZE = 16          # Reduced from 32
N_ENVS = 8              # Reduced from 16  
SIMS = 15               # Reduced from 25
USE_ENHANCED_MODEL = False  # Use basic QNet
MEMORY_EFFICIENT = True     # Enable optimizations
```

#### Option 3: Environment Variable
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 src/Train.py
```

### Performance Tips:
- Monitor console output for raster cycling
- Check `checkpoints/` folder for saved models
- Best model auto-saved as `qnet_best.pt`

## 📈 Expected Results

With the enhancements, you should see:
- **Faster convergence** (fewer episodes to good performance)
- **Better fuel-break placement** (higher rewards)
- **Stable training** (smooth loss curves)
- **Full data utilization** (all 500 rasters used)

## 🎯 Key Files Modified

- `src/Train.py` - Enhanced training with modern DQN
- `src/Model.py` - 3 neural network architectures
- `src/utils/loadingUtils.py` - Smart raster management
- `src/Env.py` - Fixed observation bug

## 📚 Documentation

- `ENHANCEMENTS.md` - Complete technical details
- `test_training.py` - Comprehensive test suite
- `simple_test.py` - Quick structure verification

---

## 🎉 You're Ready!

Your enhanced QAgent system is now ready for large-scale wildfire fuel-break placement training. The system will automatically:

✅ Load all 500 rasters  
✅ Cycle through them randomly  
✅ Use advanced DQN techniques  
✅ Save the best performing models  

**Start training with:** `python3 src/Train.py`