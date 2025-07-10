# Tensor Dimensions Fix

## Problem
The original error was:
```
RuntimeError: Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [16, 1, 8, 50, 50]
```

This occurred because the replay buffer was storing states with an extra batch dimension, resulting in 5D tensors instead of the expected 4D tensors for Conv2D layers.

## Root Cause
1. **State Preprocessing**: `preprocess_state()` returns tensors with shape `[1, 8, H, W]` (with batch dimension)
2. **Memory Storage**: States were stored in replay buffer with this batch dimension
3. **Batch Sampling**: When stacking states from replay buffer, we got `[batch_size, 1, 8, H, W]` instead of `[batch_size, 8, H, W]`

## Solution Applied

### 1. Fixed `DQNAgent.remember()` method
```python
def remember(self, state, action, reward, next_state, done):
    # Remove batch dimension if present when storing
    if state.dim() == 4 and state.size(0) == 1:
        state = state.squeeze(0)
    if next_state.dim() == 4 and next_state.size(0) == 1:
        next_state = next_state.squeeze(0)
    self.memory.push(state.cpu(), action, reward, next_state.cpu(), done)
```

### 2. Fixed `ReplayBuffer.sample()` method
```python
def sample(self, batch_size):
    # Ensure states are 3D (C, H, W) before stacking
    states_list = []
    next_states_list = []
    
    for e in experiences:
        state = e.state
        next_state = e.next_state
        
        # If state has batch dimension, remove it
        if state.dim() == 4 and state.size(0) == 1:
            state = state.squeeze(0)
        if next_state.dim() == 4 and next_state.size(0) == 1:
            next_state = next_state.squeeze(0)
            
        states_list.append(state)
        next_states_list.append(next_state)
    
    states = torch.stack(states_list)  # Now correctly (batch_size, channels, height, width)
    # ... rest of method
```

### 3. Fixed Memory Generation in `DomiRankMemoryLoader`
```python
# Store states without batch dimension
state_to_store = state_tensor.squeeze(0) if state_tensor.dim() == 4 and state_tensor.size(0) == 1 else state_tensor
next_state_to_store = next_state_tensor.squeeze(0) if next_state_tensor.dim() == 4 and next_state_tensor.size(0) == 1 else next_state_tensor

memories.append({
    'state': state_to_store.cpu(),
    'next_state': next_state_to_store.cpu(),
    # ... other fields
})
```

### 4. Fixed Training Script Memory Loading
```python
# Ensure states are tensors and have correct dimensions
state = memory['state']
next_state = memory['next_state']

if not isinstance(state, torch.Tensor):
    state = torch.tensor(state)
if not isinstance(next_state, torch.Tensor):
    next_state = torch.tensor(next_state)
    
# Add batch dimension if needed for the remember method
if state.dim() == 3:
    state = state.unsqueeze(0)
if next_state.dim() == 3:
    next_state = next_state.unsqueeze(0)
```

## Expected Tensor Flow

Now the tensor dimensions should flow correctly:

1. **Preprocessing**: `[1, 8, H, W]` → Used for inference
2. **Storage**: `[8, H, W]` → Stored in replay buffer (no batch dim)
3. **Sampling**: `[batch_size, 8, H, W]` → Correctly stacked for training
4. **Network Input**: `[batch_size, 8, H, W]` → Conv2D expects this format

## Testing

### Quick Test
```bash
python test_tensor_dimensions.py
```

This will verify that all tensor dimensions are handled correctly throughout the pipeline.

### Full Training Test
```bash
python src/scripts/train_dqn_fuel_breaks.py --config demo_config.json
```

The demo config has been updated with smaller values for faster testing:
- Grid size: 25x25 (instead of 50x50)
- Batch size: 8 (instead of 16)
- Fewer episodes and simulations

## Key Points

1. **States in memory**: Always stored as 3D tensors `[C, H, W]`
2. **States during inference**: Temporarily have batch dimension `[1, C, H, W]`
3. **States during training**: Batched as 4D tensors `[B, C, H, W]`
4. **Dimension handling**: Automatic squeezing/unsqueezing ensures compatibility

The fix maintains backward compatibility while ensuring correct tensor dimensions throughout the entire training pipeline.