# Configuration Parameters Guide

## 🔧 Complete Configuration Reference

### Fire Simulation Parameters

| Parameter | Controls | Example | Description |
|-----------|----------|---------|-------------|
| `memory_simulations` | **Number of fire simulations in `run_many_simulations`** | 50 | **This is the key parameter** that controls how many fire simulations are run each time FireEnv.step() is called |
| `fire_simulation_max_duration` | Maximum duration of each fire simulation | 120 | Time limit in minutes for each individual fire simulation |
| `num_parallel_envs` | Number of parallel fire environments | 50 | How many FireEnv instances run simultaneously |

### Environment Configuration

| Parameter | Controls | Example | Description |
|-----------|----------|---------|-------------|
| `num_landscapes` | Number of landscape data files | 50 | How many different landscape scenarios to load/generate |
| `grid_size` | Landscape grid dimensions | 50 | Size of the landscape grid (50x50) |
| `raster_dir` | Location of landscape data | "cropped_raster" | Directory containing landscape files |
| `parallel_method` | Parallelization approach | "threading" | threading/multiprocessing/sequential |
| `max_workers` | Maximum parallel workers | 25 | Number of worker threads/processes |

### Training Parameters

| Parameter | Controls | Example | Description |
|-----------|----------|---------|-------------|
| `num_episodes` | Training episodes | 50 | How many training episodes to run |
| `steps_per_episode` | Steps per episode | 30 | Environment steps per training episode |
| `train_frequency` | Training frequency | 2 | Train agent every N collected experiences |
| `batch_size` | Training batch size | 32 | Batch size for neural network training |

## 🎯 Key Parameter: `memory_simulations`

**THIS IS THE PARAMETER THAT CONTROLS `run_many_simulations` ITERATIONS**

```json
{
  "memory_simulations": 50  // <- This controls run_many_simulations(num_simulations)
}
```

### How it flows through the system:

1. **Config File** → `memory_simulations: 50`
2. **VectorizedFireEnv** → `num_simulations=50`
3. **Individual FireEnv** → `env.num_simulations = 50`
4. **FireEnv.step()** → `self.sim.run_many_simulations(50, max_duration)`

## 🚀 Performance Impact

### Memory Simulations vs Performance
- **Low values (1-10)**: Fast but unreliable rewards
- **Medium values (20-100)**: Good balance of speed and accuracy  
- **High values (100+)**: Slow but very reliable rewards

### Parallel Environments vs CPU Usage
- **1-4 envs**: Underutilizes multi-core systems
- **8-16 envs**: Good for most systems
- **25+ envs**: Requires powerful multi-core systems

## 🔍 Expected Behavior

With your config:
```json
{
  "memory_simulations": 50,
  "num_parallel_envs": 50,
  "fire_simulation_max_duration": 120
}
```

**Each training step should:**
1. Run 50 parallel environments
2. Each environment runs 50 fire simulations
3. Each simulation limited to 120 minutes max
4. Total: 2,500 fire simulations per training step

## 🛠️ Troubleshooting

### Zero Rewards Issue
- **Cause**: Fire simulations not running properly
- **Check**: Look for "acres_burned" in environment info
- **Fix**: Ensure `memory_simulations` > 0 and landscape data is valid

### Using Wrong Number of Environments  
- **Cause**: Config not loaded or landscape data limitation
- **Check**: Debug output shows "Final num_envs"
- **Fix**: Ensure config file path is correct and landscapes are generated

### CPU Not Fully Utilized
- **Cause**: `num_parallel_envs` too low or `parallel_method` = "sequential" 
- **Check**: Monitor CPU usage during training
- **Fix**: Increase `num_parallel_envs` and use "threading" method

## 📝 Recommended Settings

### For Testing (Fast)
```json
{
  "memory_simulations": 5,
  "num_parallel_envs": 4,
  "fire_simulation_max_duration": 60,
  "num_landscapes": 4
}
```

### For Production (Accurate)
```json
{
  "memory_simulations": 50,
  "num_parallel_envs": 16,
  "fire_simulation_max_duration": 120,
  "num_landscapes": 20
}
```

### For High-Performance Systems
```json
{
  "memory_simulations": 100,
  "num_parallel_envs": 32,
  "fire_simulation_max_duration": 180,
  "num_landscapes": 50
}
```