# Gym Reset Fix Summary

## Problem Description

The DQN training script was failing with the following error:

```
--- Episode 1/150 ---
Traceback (most recent call last):
  File "/home/mengsig/Projects/QNetFires/src/scripts/train_dqn_fuel_breaks_optimized.py", line 734, in <module>
    main()
  File "/home/mengsig/Projects/QNetFires/src/scripts/train_dqn_fuel_breaks_optimized.py", line 724, in main
    trainer.train()
  File "/home/mengsig/Projects/QNetFires/src/scripts/train_dqn_fuel_breaks_optimized.py", line 442, in train
    episode_results = self.collect_experiences(steps_per_episode)
  File "/home/mengsig/Projects/QNetFires/src/scripts/train_dqn_fuel_breaks_optimized.py", line 313, in collect_experiences
    observations = self.vectorized_env.reset()
  File "/home/mengsig/Projects/QNetFiresNew/qnetfires/lib/python3.10/site-packages/gym/vector/vector_env.py", line 104, in reset
    return self.reset_wait(seed=seed, options=options)
  File "/home/mengsig/Projects/QNetFiresNew/qnetfires/lib/python3.10/site-packages/gym/vector/sync_vector_env.py", line 122, in reset_wait
    observation, info = env.reset(**kwargs)
ValueError: too many values to unpack (expected 2)
```

## Root Cause

The error was caused by a mismatch between the gym API version being used (0.26.2) and the implementation of the `reset()` method in the custom environments:

1. **Gym 0.26.2** introduced a new API where `env.reset()` should return `(observation, info)` instead of just `observation`
2. **Our FireEnv** was still using the old API format, returning only the observation
3. **Gym's vectorized environment** expects the new format and tries to unpack the result into two variables: `observation, info = env.reset()`
4. **The error occurs** when Python tries to unpack 1 value (just observation) into 2 variables (observation, info)

## Solution Applied

### 1. Updated FireEnv.reset() method

**File:** `src/scripts/FireEnv.py`

**Before:**
```python
def reset(self):
    """Reset environment and return initial observation (all zeros)."""
    # Re-instantiate the simulator
    self.sim = Simulate(*self._static_args)
    self.sim.set_space_time_cubes()
    # blank observation until the run
    self.last_firemap = np.zeros((self.H, self.W), dtype=np.uint8)
    return self.last_firemap  # Only returns observation
```

**After:**
```python
def reset(self):
    """Reset environment and return initial observation (all zeros) and info."""
    # Re-instantiate the simulator
    self.sim = Simulate(*self._static_args)
    self.sim.set_space_time_cubes()
    # blank observation until the run
    self.last_firemap = np.zeros((self.H, self.W), dtype=np.uint8)
    return self.last_firemap, {}  # Returns (observation, info)
```

### 2. Updated VectorizedFireEnv to handle both formats

**File:** `src/scripts/VectorizedFireEnv.py`

Updated the vectorized environment to handle both old and new reset formats for backward compatibility:

**Before:**
```python
self.current_states = [env.reset() for env in self.envs]
```

**After:**
```python
reset_results = [env.reset() for env in self.envs]
self.current_states = [result[0] if isinstance(result, tuple) else result for result in reset_results]
```

This change was applied to:
- Main reset method (line ~97)
- Individual environment resets in step methods (lines ~153 and ~192)

### 3. Updated other training scripts

**Files updated:**
- `src/scripts/train_dqn_fuel_breaks.py` - Updated to handle new reset format
- `src/scripts/TestFuelBreaks.py` - Updated to handle new reset format

## Files Modified

1. **`src/scripts/FireEnv.py`** - Main fix: Updated reset method to return (observation, info)
2. **`src/scripts/VectorizedFireEnv.py`** - Updated to handle both old and new reset formats
3. **`src/scripts/train_dqn_fuel_breaks.py`** - Updated reset call handling
4. **`src/scripts/TestFuelBreaks.py`** - Updated reset call handling
5. **`src/scripts/train_dqn_fuel_breaks_optimized.py`** - Updated wrapper docstring (no functional change needed)

## Testing

The fix was validated with comprehensive tests that confirmed:

1. **Compatibility**: The new reset format works with gym 0.26.2's vectorized environment
2. **Backward compatibility**: The VectorizedFireEnv can handle both old and new reset formats
3. **Error reproduction**: The original error was successfully reproduced and then resolved

## Impact

- ✅ **Resolves the training crash** - The DQN training script should now run without the reset error
- ✅ **Maintains functionality** - All existing functionality is preserved
- ✅ **Future-proof** - Compatible with current gym API standards
- ✅ **Backward compatible** - VectorizedFireEnv can handle both formats during transition

## Next Steps

1. Test the training script with actual landscape data to ensure full functionality
2. Consider updating other parts of the codebase that might use older gym API patterns
3. Update documentation to reflect the new reset format requirements

## Technical Details

The error `ValueError: too many values to unpack (expected 2)` occurs when:
- Python expects: `observation, info = env.reset()`
- But receives: `just_observation` (single value)
- The unpacking fails because there's only 1 value to unpack into 2 variables

The fix ensures that `env.reset()` always returns a tuple `(observation, info)`, making it compatible with gym's vectorized environment expectations.