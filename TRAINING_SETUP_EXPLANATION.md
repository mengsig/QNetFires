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
"random_landscapes": true         // Randomly sample landscapes for each environment
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

### **🔄 NEW IMPROVED Data Usage:**
1. **Training Set**: `num_landscapes = 100` landscapes are used (16% of your 616!)
2. **Random Sampling**: Each environment randomly samples from all 100 landscapes every episode
   - Episode 1: [Landscape 45, 12, 89, 3, 67, 23, 91, 8] (8 random samples)
   - Episode 2: [Landscape 7, 56, 34, 78, 12, 90, 45, 67] (8 new random samples)
   - Episode 3: [Landscape 23, 87, 1, 56, 12, 45, 78, 34] (8 new random samples)
   - etc.

3. **Total Diversity**: Over 200 episodes × 8 environments = 1,600 random landscape samples!
4. **Evaluation**: Uses the **SAME** 100 landscapes for evaluation (still need separate test set)

### **📊 What This Means:**
- **Training Data**: 100 landscapes out of your 616 available (16% - much better!)
- **Random Sampling**: 1,600 diverse landscape experiences over training
- **Test Data**: Same 100 landscapes (still potential overfitting, but much better diversity)
- **Unused Data**: 516 landscapes unused (could increase num_landscapes further!)

---

## 🗺️ **Your 616 Maps - Are They All Used?**

### **✅ MUCH BETTER - 100 of 616 Maps Are Used!**

**NEW Usage:**
```
Available: 616 landscape maps
Used: 100 maps (16% of your data!)
Random Samples: 1,600 landscape experiences over training
Unused: 516 maps (84% still unused, but much better!)
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

**NEW IMPROVED Setup:**
- 📊 100 landscapes used for both training AND testing (16% of your data!)
- 🎲 8 parallel environments randomly sampling from these 100 landscapes each episode
- ✅ 1,600 diverse landscape experiences over 200 episodes
- 📈 516 of your 616 landscapes still unused (but much better diversity!)
- ⚠️ Still no proper train/test split (but much lower overfitting risk)

**Recommended Changes:**
- 📈 Use 400-500 landscapes for training
- 📊 Use 100+ different landscapes for testing
- 🔄 Implement proper train/validation/test splits
- 📂 Ensure your 616 maps are properly formatted and loaded

**Bottom Line**: You have amazing data (616 landscapes) but are only using a tiny fraction of it, which severely limits your model's potential!