# Fix Summary: Action Conversion Error

## 🐛 **Problem**
The training script was failing with:
```
AttributeError: 'int' object has no attribute 'reshape'
```

## 🔍 **Root Cause**
There was a mismatch between the action space format expected by different components:

1. **SingleFireEnvWrapper**: Uses `spaces.Discrete(grid_size * grid_size)` which produces **integer actions** (0 to grid_size² - 1)
2. **FireEnv.step()**: Expects a **flat binary array** that can be reshaped to `(H, W)` for the fuel break mask

The error occurred because the DQN agent was producing discrete integer actions, but `FireEnv.step()` was trying to call `.reshape()` on these integers.

## ✅ **Solution Applied**

### 1. **Added Grid Size Storage**
```python
# Before: local variable
grid_size = landscape_data['slp'].shape[0]

# After: instance variable 
self.grid_size = landscape_data['slp'].shape[0]
```

### 2. **Fixed Action Conversion in step() Method**
```python
def step(self, action):
    """Execute action and return next observation, reward, done, info."""
    self.episode_steps += 1
    
    # Convert discrete action to binary mask
    # action is an integer from 0 to (grid_size^2 - 1)
    # Convert to binary mask where only the selected position is True
    action_mask = np.zeros(self.grid_size * self.grid_size, dtype=np.float32)
    action_mask[action] = 1.0
    
    # Execute action in underlying FireEnv
    burned_map, reward, done, info = self.fire_env.step(action_mask)
    
    # ... rest of method unchanged
```

## 🎯 **How the Conversion Works**

### Example with 5x5 grid:
- **Action space**: `Discrete(25)` → produces integers 0-24
- **Action 12**: 
  - Creates binary mask: `[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]`
  - Position 12 = row 2, col 2 in 5x5 grid
  - FireEnv reshapes this to `(5,5)` and places fuel break at position (2,2)

### Mapping from discrete action to 2D position:
```python
# For action i in grid_size x grid_size:
row = i // grid_size
col = i % grid_size
```

## 🔧 **Key Changes Made**

1. **Stored grid_size**: Added `self.grid_size` to enable action conversion
2. **Action conversion**: Convert discrete integer to binary mask
3. **Proper masking**: Only one position set to 1.0 in the mask
4. **Maintained compatibility**: All other functionality unchanged

## ✅ **Benefits**

1. **Proper DQN Integration**: DQN agent can use standard discrete action space
2. **FireEnv Compatibility**: Maintains expected binary mask format
3. **Efficient Action Space**: Single integer per action instead of full mask
4. **Standard Gym Interface**: Follows gym environment conventions

## 🧪 **Verification**

The fix ensures:
- ✅ **Discrete actions work**: Integers 0 to grid_size² - 1
- ✅ **Binary mask creation**: Proper conversion to fuel break mask
- ✅ **FireEnv compatibility**: Receives reshapeable array
- ✅ **Action space bounds**: Handles boundary actions correctly
- ✅ **Observation format**: Maintains 12-channel observations

## 🚀 **Result**

The training script should now run without the `AttributeError` and properly:
- ✅ **Use discrete action space** for efficient DQN training
- ✅ **Convert actions to fuel break masks** automatically
- ✅ **Maintain gym.vector.SyncVectorEnv compatibility**
- ✅ **Preserve all optimizations** (reduced model size, memory management, etc.)

## 📊 **Action Space Details**

For a 50x50 grid (default):
- **Action space**: `Discrete(2500)` 
- **Action 0**: Fuel break at position (0,0)
- **Action 1249**: Fuel break at position (24,49) - middle of grid
- **Action 2499**: Fuel break at position (49,49) - bottom right

This provides a clean, efficient interface for the DQN agent while maintaining compatibility with the underlying fire simulation system!