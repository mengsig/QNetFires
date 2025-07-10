# Deep Q-Learning for Fuel Break Placement

This implementation provides a state-of-the-art Deep Q-Learning solution for learning optimal fuel break placement strategies in wildfire management. The system uses domirank-based expert demonstrations for initialization and learns to place fuel breaks iteratively from 1% to 20% of landscape coverage.

## Overview

The system implements an iterative approach to fuel break placement where:

1. **Domirank-based Initialization**: Uses pre-computed domirank values as expert demonstrations to initialize the Q-network's experience replay buffer
2. **Iterative Learning**: Learns to place fuel breaks progressively from 1% to 20% coverage
3. **CNN Architecture**: Employs a state-of-the-art Convolutional Neural Network with residual connections and spatial attention
4. **Fire Simulation Integration**: Uses the existing fire simulation environment (`run_many_simulations`) for realistic reward evaluation

## Key Features

### 🧠 **State-of-the-Art CNN Architecture**
- **Residual Connections**: Deep residual blocks for improved gradient flow
- **Spatial Attention**: Attention mechanism to focus on important landscape features
- **Multi-scale Features**: Progressive feature extraction from 64 to 512 channels
- **Batch Normalization**: For training stability and faster convergence

### 🎯 **Domirank-based Expert Initialization**
- Loads existing domirank fuel break patterns as expert demonstrations
- Converts domirank strategies into experience replay memories
- Supports iterative fuel break percentages (1%, 2%, ..., 20%)
- Pre-trains the network on expert knowledge before exploration

### 🔄 **Iterative Fuel Break Placement**
- Progressive learning from 1% to 20% fuel break coverage
- Prevents fuel break overlap through action masking
- Supports both small-scope (immediate) and large-scope (final) rewards
- Configurable percentage increments for fine-grained control

### 🔥 **Fire Simulation Integration**
- Uses `run_many_simulations` for robust reward evaluation
- Integrates with existing `Simulate` and `FireEnv` classes
- Evaluates fuel break effectiveness through multiple random fire starts
- Provides realistic feedback for learning optimal strategies

## Installation

### Prerequisites
```bash
# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy matplotlib rasterio gym tqdm scikit-learn
```

### Repository Setup
```bash
# Ensure you have the existing fire simulation code
# The implementation expects the following structure:
# src/
#   scripts/
#     Simulate.py
#     FireEnv.py
#   utils/
#     loadingUtils.py
#     networkUtils.py
#     plottingUtils.py
```

## Quick Start

### 1. Demo Run
```bash
# Test the implementation with a simple demo
python demo_dqn_fuel_breaks.py
```

This will:
- Generate sample landscape data
- Test all components
- Create visualizations
- Run a simple training loop

### 2. Full Training
```bash
# Train with demo configuration (50 episodes)
python src/scripts/train_dqn_fuel_breaks.py --config demo_config.json

# Train with default configuration (1000 episodes)
python src/scripts/train_dqn_fuel_breaks.py --num_episodes 1000 --num_landscapes 10
```

### 3. Evaluation Only
```bash
# Evaluate a trained model
python src/scripts/train_dqn_fuel_breaks.py --evaluate_only outputs/models/best_model.pt
```

## Configuration

### Configuration File Format
```json
{
  "raster_dir": "cropped_raster",
  "grid_size": 50,
  "input_channels": 8,
  "num_landscapes": 5,
  
  "num_episodes": 1000,
  "max_fuel_break_percentage": 20.0,
  "percentage_increment": 1.0,
  "memory_simulations": 5,
  "pretrain_steps": 1000,
  
  "learning_rate": 1e-4,
  "gamma": 0.95,
  "initial_epsilon": 1.0,
  "epsilon_min": 0.01,
  "epsilon_decay": 0.995,
  "buffer_size": 100000,
  "batch_size": 32,
  
  "target_update_frequency": 100,
  "checkpoint_frequency": 100,
  
  "output_dir": "outputs",
  "regenerate_memories": false
}
```

### Key Parameters

#### Environment Settings
- `raster_dir`: Directory containing landscape raster data
- `grid_size`: Spatial resolution of landscape grids (e.g., 50x50)
- `input_channels`: Number of landscape layers (slp, asp, dem, cc, cbd, cbh, ch, fbfm)
- `num_landscapes`: Number of different landscape scenarios

#### Training Settings
- `num_episodes`: Total training episodes
- `max_fuel_break_percentage`: Maximum fuel break coverage (default: 20%)
- `percentage_increment`: Step size for iterative placement (default: 1%)
- `memory_simulations`: Fire simulations per domirank memory evaluation
- `pretrain_steps`: Training steps on domirank memories before exploration

#### Agent Settings
- `learning_rate`: Adam optimizer learning rate
- `gamma`: Q-learning discount factor
- `epsilon_*`: Exploration schedule parameters
- `buffer_size`: Experience replay buffer capacity
- `batch_size`: Training batch size

## Architecture Details

### Network Architecture
```
Input: (batch_size, 8, height, width)
├── Conv2d(8→64) + BN + ReLU
├── Conv2d(64→128) + BN + ReLU
├── ResidualBlock(128→128)
├── ResidualBlock(128→256)
├── ResidualBlock(256→256)
├── SpatialAttention(256)
├── Conv2d(256→512) + BN + ReLU + Dropout
├── Conv2d(512→256) + BN + ReLU
├── Conv2d(256→128) + BN + ReLU
└── Conv2d(128→1) → Q-values(height×width)
```

### Residual Block
```
Input → Conv2d + BN + ReLU → Conv2d + BN → (+) → ReLU
  ↓                                        ↑
  Skip Connection ────────────────────────┘
```

### Spatial Attention
```
Input(C,H,W) → Conv2d(C→C/8) + ReLU → Conv2d(C/8→1) + Sigmoid → Attention Map
     ↓                                                              ↑
     Element-wise Multiplication ←──────────────────────────────────┘
```

## Data Format

### Landscape Data
The system expects landscape data in the following format:
```python
landscape_data = {
    'slp': np.array,    # Slope (degrees)
    'asp': np.array,    # Aspect (degrees)
    'dem': np.array,    # Elevation (meters)
    'cc': np.array,     # Canopy cover (%)
    'cbd': np.array,    # Canopy bulk density
    'cbh': np.array,    # Canopy base height
    'ch': np.array,     # Canopy height
    'fbfm': np.array    # Fuel model categories
}
```

### Domirank Data
Domirank values should be stored as text files:
```
cropped_raster/
└── domirank/
    ├── domirank_0.txt
    ├── domirank_1.txt
    └── ...
```

Each file contains flattened domirank values that can be reshaped to (grid_size, grid_size).

## Training Process

### 1. Initialization Phase
1. **Data Preparation**: Load or generate landscape data and domirank values
2. **Memory Generation**: Convert domirank fuel breaks into experience replay memories
3. **Pre-training**: Train the Q-network on domirank memories for initialization

### 2. Training Phase
1. **Episode Loop**: For each training episode:
   - Select a random landscape
   - Initialize with no fuel breaks
   - Iteratively place fuel breaks from 1% to 20%
   - Use epsilon-greedy action selection
   - Evaluate rewards using fire simulation
   - Store experiences and train the network

### 3. Evaluation Phase
1. **Performance Assessment**: Evaluate trained models on held-out landscapes
2. **Comparison**: Compare against domirank baseline and random placement
3. **Visualization**: Generate fuel break placement visualizations

## Outputs

### Model Files
- `best_model.pt`: Best performing model during training
- `final_model.pt`: Final model after all training episodes
- `checkpoint_episode_*.pt`: Periodic training checkpoints

### Metrics and Logs
- `training_metrics.json`: Episode rewards, losses, and statistics
- `training_progress.png`: Training curves and performance plots

### Visualizations
- Fuel break placement patterns at different percentages
- Training progress curves
- Landscape feature importance maps

## Usage Examples

### Basic Training
```python
from src.scripts.train_dqn_fuel_breaks import IterativeFuelBreakTrainer

config = {
    'num_episodes': 500,
    'num_landscapes': 10,
    'grid_size': 50,
    # ... other parameters
}

trainer = IterativeFuelBreakTrainer(config)
trainer.train()
```

### Custom Memory Generation
```python
from src.scripts.DomiRankMemoryLoader import DomiRankMemoryLoader

loader = DomiRankMemoryLoader(grid_size=50)
memories = loader.generate_training_memories(
    landscape_indices=[0, 1, 2],
    percentages=list(range(1, 21)),
    num_simulations=10
)
```

### Model Evaluation
```python
from src.scripts.DQNAgent import DQNAgent

agent = DQNAgent(grid_size=50)
agent.load_model('outputs/models/best_model.pt')

# Evaluate on new landscape
landscape_data = load_landscape(index=5)
state = agent.preprocess_state(landscape_data)
action = agent.act(state)  # Get fuel break location
```

## Performance Optimization

### Hardware Recommendations
- **CPU**: Multi-core processor for fire simulations
- **GPU**: CUDA-compatible GPU for neural network training (optional)
- **Memory**: 8GB+ RAM for larger landscapes and replay buffers

### Training Optimizations
- Use smaller grid sizes (25x25) for faster prototyping
- Reduce `memory_simulations` for faster memory generation
- Use gradient accumulation for larger effective batch sizes
- Enable mixed precision training for GPU acceleration

### Hyperparameter Tuning
- **Learning Rate**: Start with 1e-4, adjust based on convergence
- **Epsilon Decay**: Slower decay (0.999) for more exploration
- **Buffer Size**: Larger buffers (100K+) for better sample diversity
- **Batch Size**: Powers of 2 (16, 32, 64) for optimal GPU utilization

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce `grid_size` or `buffer_size`
   - Use smaller batch sizes
   - Generate fewer training memories

2. **Slow Training**
   - Reduce `memory_simulations`
   - Use smaller landscapes for prototyping
   - Enable GPU acceleration

3. **Poor Convergence**
   - Increase pre-training steps
   - Adjust learning rate
   - Check reward scaling

4. **Missing Dependencies**
   ```bash
   # Install missing packages
   pip install -r requirements.txt
   ```

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use smaller configuration for testing
config['num_episodes'] = 10
config['grid_size'] = 25
config['memory_simulations'] = 2
```

## Contributing

To extend this implementation:

1. **Custom Reward Functions**: Modify reward calculation in `evaluate_fuel_break_performance`
2. **Network Architectures**: Experiment with different CNN designs in `DQNNetwork`
3. **Action Spaces**: Implement continuous or hierarchical action spaces
4. **Multi-objective Optimization**: Add multiple reward components (cost, effectiveness, accessibility)

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{dqn_fuel_breaks_2024,
  title={Deep Q-Learning for Iterative Fuel Break Placement},
  author={AI Assistant},
  year={2024},
  url={https://github.com/your-repo/dqn-fuel-breaks}
}
```

## License

This implementation is provided under the MIT License. See LICENSE file for details.

---

For questions or issues, please open a GitHub issue or contact the development team.