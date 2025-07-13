# 🎯 **Complete Training Setup Explanation**

## 📋 **Configuration Parameters Explained**

### **🏞️ Environment & Data Parameters**
```json
"raster_dir": "cropped_raster"     // Directory containing landscape raster files
"grid_size": 50                    // Spatial resolution: 50x50 grid cells  
"input_channels": 12               // Number of landscape data layers (see below)
"num_landscapes": 4                // How many different landscapes to use for training
```

**Input Channels (12 total):**
1. `slp` - Slope (degrees)
2. `asp` - Aspect (degrees) 
3. `dem` - Digital Elevation Model (meters)
4. `cc` - Canopy Cover (percentage)
5. `cbd` - Canopy Bulk Density
6. `cbh` - Canopy Base Height
7. `ch` - Canopy Height
8. `fbfm` - Fuel Model Categories (1-14)
9. `fireline_north` - Fire intensity northward
10. `fireline_east` - Fire intensity eastward
11. `fireline_south` - Fire intensity southward
12. `fireline_west` - Fire intensity westward

### **🚀 Training Parameters**
```json
"num_episodes": 200                // Total training episodes to run
"steps_per_episode": 50           // Environment steps per episode
"num_parallel_envs": 8            // Number of environments running simultaneously
"parallel_method": "threading"    // Use threading for parallelization
"train_frequency": 4              // Train neural network every 4 environment steps
```

### **🧠 Neural Network Parameters**
```json
"learning_rate": 1e-4             // Adam optimizer learning rate
"gamma": 0.95                     // Discount factor for future rewards
"initial_epsilon": 1.0            // Start with 100% exploration
"epsilon_min": 0.01               // Minimum exploration rate (1%)
"epsilon_decay": 0.995            // Decay exploration by 0.5% each training step
"buffer_size": 25000              // Replay buffer size (experiences stored)
"batch_size": 32                  // Mini-batch size for neural network training
```

### **💾 Memory Management Parameters**
```json
"experience_buffer_size": 1000    // Local experience buffer (before replay buffer)
"max_history_size": 500           // Maximum metrics kept in memory
"cleanup_frequency": 1000         // Clean GPU memory every 1000 training steps
"memory_cleanup_frequency": 5     // Clean episode memory every 5 episodes
```

### **🔥 Fire Simulation Parameters**
```json
"memory_simulations": 3           // Fire simulations run per training step
"fire_simulation_max_duration": 30 // Maximum simulation time (minutes)
"pretrain_steps": 200             // Pre-training steps on expert demonstrations
"percentage_increment": 2         // DomiRank percentage increments (1%, 3%, 5%...)
```

### **📊 Training Schedule Parameters**
```json
"target_update_frequency": 20     // Update target network every 20 episodes
"checkpoint_frequency": 20        // Save model checkpoint every 20 episodes
"eval_steps_per_episode": 30      // Steps per episode during evaluation
```

---

## 🎯 **Your Training vs Testing Setup**

### **❌ NO EXPLICIT TRAIN/TEST SPLIT**
Your current setup does **NOT** have a traditional train/test split! Here's what actually happens:

### **🔄 Current Data Usage:**
1. **Training Set**: `num_landscapes = 4` landscapes are used
2. **Cycling**: The 8 parallel environments cycle through these 4 landscapes
   - Environment 0 → Landscape 0
   - Environment 1 → Landscape 1  
   - Environment 2 → Landscape 2
   - Environment 3 → Landscape 3
   - Environment 4 → Landscape 0 (cycles back)
   - Environment 5 → Landscape 1
   - etc.

3. **Evaluation**: Uses the **SAME** 4 landscapes for evaluation (no separate test set)

### **📊 What This Means:**
- **Training Data**: Only 4 landscapes out of your 616 available
- **Test Data**: Same 4 landscapes (potential overfitting!)
- **Unused Data**: 612 landscapes are completely unused!

---

## 🗺️ **Your 616 Maps - Are They All Used?**

### **❌ NO - Only 4 of 616 Maps Are Used!**

**Current Usage:**
```
Available: 616 landscape maps
Used: 4 maps (0.65% of your data!)
Unused: 612 maps (99.35% wasted!)
```

**Why So Few?**
- `num_landscapes: 4` limits the system to only 4 landscapes
- The system generates SYNTHETIC data if the `cropped_raster` directory doesn't exist
- Your 616 real maps are likely not in the expected format/location

---

## 📈 **Recommendations for Better Training**

### **1. 🎯 Use More of Your Data**
```json
"num_landscapes": 500    // Use 500 of your 616 landscapes for training
```

### **2. 📊 Implement Proper Train/Test Split**
```json
"num_train_landscapes": 400     // 400 landscapes for training
"num_test_landscapes": 100      // 100 landscapes for testing
"num_val_landscapes": 50        // 50 landscapes for validation
```

### **3. 🔄 Proper Data Loading**
Ensure your 616 maps are in the correct format:
```
cropped_raster/
├── slp/
│   ├── landscape_0_slp.npy
│   ├── landscape_1_slp.npy
│   └── ... (616 files)
├── asp/
│   ├── landscape_0_asp.npy
│   └── ... (616 files)
└── ... (for each of the 8 landscape types)
```

### **4. 📋 Better Configuration**
```json
{
  "num_landscapes": 500,
  "num_parallel_envs": 16,        // More environments for more diversity
  "train_test_split": 0.8,        // 80% train, 20% test
  "landscape_rotation": true,      // Rotate through all landscapes
  "evaluation_landscapes": "separate"  // Use different landscapes for eval
}
```

---

## 🚨 **Current Issues to Fix**

### **1. Data Underutilization**
- You're only using 0.65% of your landscape data
- This severely limits the model's ability to generalize

### **2. No Test Set**
- Training and evaluation use the same 4 landscapes
- High risk of overfitting
- No way to measure true generalization performance

### **3. Synthetic vs Real Data**
- If `cropped_raster` doesn't exist, system generates fake data
- Your 616 real landscapes might not be properly loaded

### **4. Limited Diversity**
- Only 4 landscapes means limited environmental diversity
- Agent learns very specific patterns instead of general strategies

---

## 🎯 **Summary**

**Current Setup:**
- 📊 4 landscapes used for both training AND testing
- 🔄 8 parallel environments cycling through these 4 landscapes  
- ❌ 612 of your 616 landscapes are unused
- ⚠️ No proper train/test split (high overfitting risk)

**Recommended Changes:**
- 📈 Use 400-500 landscapes for training
- 📊 Use 100+ different landscapes for testing
- 🔄 Implement proper train/validation/test splits
- 📂 Ensure your 616 maps are properly formatted and loaded

**Bottom Line**: You have amazing data (616 landscapes) but are only using a tiny fraction of it, which severely limits your model's potential!