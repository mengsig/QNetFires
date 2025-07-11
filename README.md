# FireBreak Q-Network (QNetFires) 🔥

A sophisticated deep reinforcement learning system for optimizing fuel break placement in wildfire management. This project uses Deep Q-Networks (DQN) to learn optimal fuel break strategies by analyzing landscape characteristics and fire behavior patterns.

## Overview

QNetFires combines landscape analysis, fire behavior modeling, and deep reinforcement learning to intelligently place fuel breaks that minimize wildfire damage. The system analyzes 12 different landscape and fire behavior channels to make informed decisions about where to place fuel breaks for maximum effectiveness.

## Key Features

- **Deep Q-Network Architecture**: State-of-the-art CNN with residual blocks and attention mechanisms
- **Comprehensive Landscape Analysis**: 12-channel input including traditional landscape features and directional fireline intensity data
- **Parallel Training**: Vectorized environments for efficient training across multiple landscapes
- **DomiRank Integration**: Expert demonstrations from network-based centrality analysis
- **Fire Behavior Modeling**: Integration with pyretechnics for realistic fire simulation
- **Visualization Tools**: Comprehensive plotting and analysis capabilities

## Input Channels (12 Total)

The system analyzes 12 different landscape and fire behavior characteristics:

### Traditional Landscape Features (8 channels):
1. **Slope (slp)**: Terrain slope in degrees
2. **Aspect (asp)**: Terrain aspect (compass direction)
3. **Elevation (dem)**: Digital elevation model
4. **Canopy Cover (cc)**: Percentage of tree canopy coverage
5. **Canopy Bulk Density (cbd)**: Density of canopy material
6. **Canopy Base Height (cbh)**: Height of the tree canopy base
7. **Canopy Height (ch)**: Total height of the tree canopy
8. **Fuel Model (fbfm)**: Fire behavior fuel model classification

### Fireline Intensity Features (4 channels):
9. **North Fireline Intensity**: Fire intensity spreading northward
10. **East Fireline Intensity**: Fire intensity spreading eastward
11. **South Fireline Intensity**: Fire intensity spreading southward
12. **West Fireline Intensity**: Fire intensity spreading westward

These fireline intensity channels provide crucial information about fire behavior in different directions, enabling more precise fuel break placement.

## Installation

### Quick Install (Arch Linux)
```bash
bash install.sh --IUseArchBtw
```

### Quick Install (Ubuntu)
```bash install.sh
```

### Manual Installation (All Systems)
Ensure you have **Python 3.10** installed, then:

```bash
git clone git@github.com:pyregence/pyretechnics.git
git clone git@github.com:mengsig/DomiRank.git
python -m venv qnetfires
source qnetfires/bin/activate
pip install -e DomiRank/.
pip install -r requirements.txt
cd pyretechnics
python setup.py install
cd ..
```

## Usage

### Quick Start
The `run.sh` script provides an easy way to run complete training pipelines:

```bash
bash run.sh
```

### Configuration

Edit the key parameters in `parallel_config.json`:

```json
{
  "raster_dir": "cropped_raster",
  "grid_size": 50,
  "input_channels": 12,
  "num_landscapes": 8,
  "num_episodes": 200,
  "num_parallel_envs": 8,
  "learning_rate": 1e-4,
  "batch_size": 32
}
```

### Training Pipeline

1. **Generate Landscape Data**: Create or load landscape raster data
2. **Compute Adjacency**: Run `CreateAdjacency.py` to generate fireline intensity data
3. **Train Agent**: Use parallel training with `train_dqn_fuel_breaks_parallel.py`
4. **Visualize Results**: View results with `visualize_agent_fuel_breaks.py`

## Architecture

### DQN Network
- **Input**: 12-channel landscape tensor (batch_size, 12, height, width)
- **Architecture**: CNN with residual blocks and spatial attention
- **Output**: Q-values for each spatial location (fuel break placement)

### Training Process
1. **Experience Collection**: Parallel environments generate diverse experiences
2. **Expert Demonstrations**: DomiRank provides high-quality initial experiences
3. **Iterative Learning**: Agent learns from both exploration and expert guidance
4. **Reward Signal**: Based on fire damage reduction from fuel breaks

## Key Components

### Core Scripts
- `CreateAdjacency.py`: Generates fireline intensity data for all directions
- `DQNAgent.py`: Deep Q-Network implementation with modern architecture
- `VectorizedFireEnv.py`: Parallel environment wrapper for efficient training
- `train_dqn_fuel_breaks_parallel.py`: Main training loop with parallel execution

### Utilities
- `loadingUtils.py`: Landscape and fireline data loading
- `DomiRankMemoryLoader.py`: Expert demonstration generation
- `FireEnv.py`: Single environment wrapper
- `Simulate.py`: Fire behavior simulation

## Configuration Guide

| Parameter | Description | Default |
|-----------|-------------|---------|
| `input_channels` | Number of input channels (landscape + fireline) | 12 |
| `grid_size` | Spatial resolution of landscape grid | 50 |
| `num_landscapes` | Number of different landscapes for training | 8 |
| `num_episodes` | Total training episodes | 200 |
| `num_parallel_envs` | Parallel environments (≤ num_landscapes) | 8 |
| `learning_rate` | DQN learning rate | 1e-4 |
| `batch_size` | Training batch size | 32 |
| `experience_buffer_size` | Experience replay buffer size | 1000 |
| `steps_per_episode` | Fire simulations per episode | 100 |

## Data Structure

```
cropped_raster/
├── slp/           # Slope data
├── asp/           # Aspect data
├── dem/           # Elevation data
├── cc/            # Canopy cover data
├── cbd/           # Canopy bulk density data
├── cbh/           # Canopy base height data
├── ch/            # Canopy height data
├── fbfm/          # Fuel model data
├── fireline/      # Fireline intensity data
│   ├── fireline_north_{k}.txt
│   ├── fireline_east_{k}.txt
│   ├── fireline_south_{k}.txt
│   └── fireline_west_{k}.txt
└── domirank/      # DomiRank centrality data
    └── domirank_{k}.txt
```

## Performance Features

- **GPU Acceleration**: Automatic CUDA detection and optimization
- **Parallel Training**: Multi-environment vectorized execution
- **Memory Management**: Efficient experience replay and buffer management
- **Gradient Clipping**: Stable training with gradient normalization

## Research Applications

This system is designed for:
- **Wildfire Management**: Optimal fuel break placement strategies
- **Landscape Planning**: Fire risk assessment and mitigation
- **Emergency Response**: Rapid fuel break deployment planning
- **Research**: Fire behavior modeling and landscape analysis

## Contributing

This project integrates multiple advanced techniques:
- Deep reinforcement learning (DQN)
- Fire behavior modeling (pyretechnics)
- Network analysis (DomiRank)
- Parallel computing
- Landscape analysis

Results are saved in `src/results/<experiment_name>/` for analysis and comparison.

## License

This project combines multiple open-source components. Please refer to individual component licenses for details.

