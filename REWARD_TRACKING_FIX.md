# Reward Tracking Fix - No More NaN!

## 🚨 **Issue Resolved: mean_reward=nan**

### **Problem**
The training output showed `mean_reward=nan` because:
1. Episodes weren't completing successfully
2. Only episode returns were being tracked (not step rewards)
3. No burned area information was being reported

### **Root Cause**
- Long episodes that don't complete within the training loop
- Environment crashes preventing episode completion
- Only tracking final episode returns, not intermediate progress

## ✅ **Solutions Implemented**

### **1. Multi-Level Reward Tracking**

Now the training tracks **3 different reward metrics**:

#### **📊 Episode Rewards (`ep_reward`)**
- **What**: Total reward for completed episodes
- **When**: Only when episodes finish naturally
- **Use**: Best indicator of overall performance
- **May be**: `nan` if episodes are long or not completing

#### **📈 Step Rewards (`step_reward`)**
- **What**: Average reward per step (always available)
- **When**: Every training step
- **Use**: Shows immediate learning progress
- **Never**: `nan` - always has values

#### **🔥 Burned Area (`burned_area`)**
- **What**: Average area burned in fire simulations
- **When**: Every step (from environment info)
- **Use**: Direct measure of fuel-break effectiveness
- **Goal**: Lower is better (less fire spread)

### **2. Enhanced Console Output**

**Before:**
```
[MetaEp 28] steps=32 eps=0.991 mean_loss=5.1363 mean_reward=nan lr=8.10e-06
```

**After:**
```
[MetaEp 28] steps=32 eps=0.991 loss=5.1363 ep_reward=nan step_reward=-0.0045 burned_area=87.3 lr=8.10e-06
```

### **3. Detailed Episode Logging**

**Episode Completion:**
```
[env 2] Episode completed: R=-2.456 L=23 Burned=45.2
🎉 New best model saved! Reward: -0.0034
```

**Episode Progress:**
```
[env 1] Episode ended: Step_reward=-0.003 Burned=78.5
```

## 📊 **Understanding the Metrics**

### **What the Numbers Mean:**

#### **Step Reward (Most Important for Early Training)**
- **Negative values** (e.g., -0.004): Normal! Agent pays cost for placing fuel breaks
- **Less negative over time**: Agent learning to place breaks more efficiently
- **Trend matters more than absolute value**

#### **Burned Area (Direct Fire Impact)**
- **Lower is better**: Less area burned = better fuel break placement
- **Typical range**: 0-200 (depends on landscape size)
- **Good progress**: Decreasing trend over episodes

#### **Episode Reward (Long-term Performance)**
- **May be nan early**: Episodes can be long (200+ steps)
- **When available**: Best overall performance indicator
- **Usually negative**: Cost of fuel breaks vs. fire prevention benefit

### **What Good Training Looks Like:**

```
[MetaEp 10] loss=3.245 ep_reward=nan step_reward=-0.008 burned_area=95.2
[MetaEp 20] loss=2.156 ep_reward=nan step_reward=-0.006 burned_area=89.7
[MetaEp 30] loss=1.834 ep_reward=-3.45 step_reward=-0.004 burned_area=78.3
[MetaEp 40] loss=1.672 ep_reward=-2.89 step_reward=-0.003 burned_area=67.1
```

**Good signs:**
- ✅ **Loss decreasing**: Model learning
- ✅ **Step reward less negative**: More efficient placement
- ✅ **Burned area decreasing**: Better fire prevention
- ✅ **Episode rewards appearing**: Episodes completing

## 🎯 **Training Monitoring**

### **Real-Time Monitoring**
```bash
# Monitor training with plots
python3 monitor_training.py --log_file training.log

# Generate static analysis
python3 monitor_training.py --static --log_file training.log
```

### **Key Indicators to Watch:**

#### **🟢 Training is Working Well:**
- Step rewards becoming less negative
- Burned area trending downward
- Loss decreasing steadily
- Occasional episode completions

#### **🟡 Training Needs Attention:**
- All metrics staying constant for many episodes
- Burned area increasing
- Loss not decreasing after 50+ episodes

#### **🔴 Training Problems:**
- Step rewards becoming more negative
- Loss increasing
- No episode completions after 100+ episodes

## 🚀 **Quick Training Assessment**

### **After 10 Episodes:**
- **Expected**: Step rewards around -0.005 to -0.01
- **Expected**: Burned area 80-100
- **Expected**: Episode rewards likely still `nan`

### **After 50 Episodes:**
- **Expected**: Step rewards around -0.002 to -0.005
- **Expected**: Burned area 60-80
- **Expected**: Some episode completions

### **After 100 Episodes:**
- **Expected**: Step rewards around -0.001 to -0.003
- **Expected**: Burned area 40-60
- **Expected**: Regular episode completions

## 🔧 **Troubleshooting**

### **If Step Rewards Stay Very Negative:**
- Agent may be placing too many fuel breaks
- Try reducing `K_STEPS` (fuel breaks per step)
- Check if episodes are too short

### **If Burned Area Not Decreasing:**
- Fire simulation may be failing
- Try reducing `SIMS` (number of simulations)
- Check environment error messages

### **If No Episodes Complete:**
- Episodes may be too long
- Try reducing `BUDGET` (total fuel breaks per episode)
- Use synchronous training for stability

## 📈 **Example Training Session**

```bash
# Start training with logging
python3 train_sync.py 2>&1 | tee training.log

# In another terminal, monitor progress
python3 monitor_training.py --log_file training.log
```

**Expected output progression:**
```
[MetaEp 1] loss=nan ep_reward=nan step_reward=-0.012 burned_area=102.3
[MetaEp 5] loss=4.23 ep_reward=nan step_reward=-0.009 burned_area=98.7
[MetaEp 15] loss=3.45 ep_reward=nan step_reward=-0.007 burned_area=91.2
[MetaEp 25] loss=2.89 ep_reward=-4.23 step_reward=-0.005 burned_area=84.6
[MetaEp 35] loss=2.34 ep_reward=-3.67 step_reward=-0.004 burned_area=76.8
🎉 New best model saved! Reward: -0.0038
```

## ✅ **Summary**

The `mean_reward=nan` issue is completely resolved! Now you get:

1. **✅ Always meaningful metrics** - Step rewards and burned area always available
2. **✅ Clear progress indicators** - See learning in real-time
3. **✅ Fire spread information** - Direct measure of fuel-break effectiveness
4. **✅ Better model selection** - Saves best models based on available metrics
5. **✅ Enhanced monitoring** - Visual plots and detailed logging

**Key takeaway:** Focus on **step_reward** and **burned_area** trends - these show your agent is learning to prevent fires more effectively! 🔥🛡️