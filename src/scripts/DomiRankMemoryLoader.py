import numpy as np
import os
import sys
import torch
from typing import List, Tuple, Dict, Any
from collections import defaultdict
import matplotlib.pyplot as plt

# Add src to path for imports
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.insert(0, project_root)

from src.utils.loadingUtils import load_all_rasters
from src.scripts.Simulate import Simulate


class DomiRankMemoryLoader:
    """
    Loads domirank-based fuel breaks and converts them into training memories
    for Deep Q-learning initialization.
    
    This class implements iterative domirank fuel break placement at different
    percentages (1% to 20%) and creates corresponding experience memories.
    """
    
    def __init__(self, raster_dir="cropped_raster", grid_size=50):
        self.raster_dir = raster_dir
        self.grid_size = grid_size
        self.memories = []
        self.landscape_indices = []
        
        # Setup directories
        self.domirank_dir = os.path.join(raster_dir, "domirank")
        self.ensure_directories()
        
    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        os.makedirs(self.raster_dir, exist_ok=True)
        os.makedirs(self.domirank_dir, exist_ok=True)
        
        # Create subdirectories for each landscape type
        landscape_types = ['slp', 'asp', 'dem', 'cc', 'cbd', 'cbh', 'ch', 'fbfm']
        for landscape_type in landscape_types:
            os.makedirs(os.path.join(self.raster_dir, landscape_type), exist_ok=True)
    
    def generate_sample_data(self, num_landscapes=5):
        """
        Generate sample landscape data for testing purposes.
        In a real scenario, this would be replaced with actual landscape data.
        """
        print(f"Generating {num_landscapes} sample landscapes...")
        
        landscape_types = {
            'slp': (0, 45),      # slope in degrees
            'asp': (0, 360),     # aspect in degrees 
            'dem': (0, 3000),    # elevation in meters
            'cc': (0, 100),      # canopy cover percentage
            'cbd': (0, 50),      # canopy bulk density
            'cbh': (0, 30),      # canopy base height
            'ch': (0, 60),       # canopy height
            'fbfm': (1, 14)      # fuel model categories
        }
        
        for i in range(num_landscapes):
            # Create landscape data
            landscape_data = {}
            for name, (min_val, max_val) in landscape_types.items():
                if name == 'fbfm':
                    # Discrete fuel model values
                    data = np.random.randint(min_val, max_val + 1, 
                                          size=(self.grid_size, self.grid_size))
                else:
                    # Continuous values with some spatial correlation
                    data = self._generate_correlated_field(min_val, max_val)
                landscape_data[name] = data
                
                # Save individual landscape files
                landscape_dir = os.path.join(self.raster_dir, name)
                filepath = os.path.join(landscape_dir, f"{self.raster_dir}_{i}_{name}.npy")
                np.save(filepath, data)
            
            # Generate domirank values for this landscape
            domirank_values = self._generate_domirank_values(landscape_data)
            domirank_filepath = os.path.join(self.domirank_dir, f"domirank_{i}.txt")
            np.savetxt(domirank_filepath, domirank_values)
            
            print(f"Generated landscape {i}")
    
    def _generate_correlated_field(self, min_val, max_val):
        """Generate a spatially correlated random field."""
        # Start with random noise
        field = np.random.random((self.grid_size, self.grid_size))
        
        # Apply Gaussian smoothing for spatial correlation
        from scipy.ndimage import gaussian_filter
        field = gaussian_filter(field, sigma=2.0)
        
        # Normalize to desired range
        field = (field - field.min()) / (field.max() - field.min())
        field = field * (max_val - min_val) + min_val
        
        return field.astype(np.float32)
    
    def _generate_domirank_values(self, landscape_data):
        """
        Generate synthetic domirank values based on landscape characteristics.
        Higher values indicate better fuel break locations.
        """
        # Combine multiple landscape factors to create domirank-like values
        slope = landscape_data['slp']
        elevation = landscape_data['dem']
        canopy_cover = landscape_data['cc']
        
        # Higher domirank for:
        # - Moderate slopes (easier access)
        # - Ridge lines (higher elevation locally)
        # - Lower canopy cover (easier to create breaks)
        
        # Normalize inputs
        slope_norm = (slope - slope.min()) / (slope.max() - slope.min() + 1e-8)
        elev_norm = (elevation - elevation.min()) / (elevation.max() - elevation.min() + 1e-8)
        cc_norm = (canopy_cover - canopy_cover.min()) / (canopy_cover.max() - canopy_cover.min() + 1e-8)
        
        # Composite score (higher is better for fuel breaks)
        domirank = (
            0.3 * (1 - np.abs(slope_norm - 0.3)) +  # Prefer moderate slopes
            0.4 * elev_norm +                        # Prefer higher elevations
            0.3 * (1 - cc_norm)                      # Prefer lower canopy cover
        )
        
        # Add some noise
        domirank += 0.1 * np.random.random(domirank.shape)
        
        return domirank.flatten()
    
    def load_landscape_data(self, index: int) -> Dict[str, np.ndarray]:
        """Load landscape data for a given index."""
        try:
            # Try to load from files first
            raster_dict = load_all_rasters(self.raster_dir, index)
            return raster_dict
        except:
            # If files don't exist, try loading from numpy files
            landscape_data = {}
            landscape_types = ['slp', 'asp', 'dem', 'cc', 'cbd', 'cbh', 'ch', 'fbfm']
            
            for name in landscape_types:
                filepath = os.path.join(self.raster_dir, name, f"{self.raster_dir}_{index}_{name}.npy")
                if os.path.exists(filepath):
                    landscape_data[name] = np.load(filepath)
                else:
                    # Generate if doesn't exist
                    print(f"Warning: {filepath} not found. Generating sample data...")
                    if not hasattr(self, '_sample_generated'):
                        self.generate_sample_data(max(5, index + 1))
                        self._sample_generated = True
                    landscape_data[name] = np.load(filepath)
            
            return landscape_data
    
    def load_domirank_values(self, index: int) -> np.ndarray:
        """Load domirank values for a given landscape index."""
        filepath = os.path.join(self.domirank_dir, f"domirank_{index}.txt")
        
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found. Generating sample data...")
            if not hasattr(self, '_sample_generated'):
                self.generate_sample_data(max(5, index + 1))
                self._sample_generated = True
        
        domirank_values = np.loadtxt(filepath)
        return domirank_values.reshape(self.grid_size, self.grid_size)
    
    def create_iterative_fuel_breaks(self, domirank_values: np.ndarray, 
                                   percentages: List[float]) -> Dict[float, np.ndarray]:
        """
        Create iterative fuel breaks at different percentages based on domirank values.
        
        Args:
            domirank_values: 2D array of domirank values
            percentages: List of percentages (e.g., [1, 2, 3, ..., 20])
            
        Returns:
            Dictionary mapping percentage to fuel break mask
        """
        fuel_breaks = {}
        total_cells = domirank_values.size
        
        # Sort domirank values in descending order (highest values first)
        flat_domirank = domirank_values.flatten()
        sorted_indices = np.argsort(flat_domirank)[::-1]
        
        for percentage in percentages:
            num_breaks = int(total_cells * percentage / 100.0)
            
            # Create binary mask for fuel breaks
            fuel_break_mask = np.zeros_like(flat_domirank, dtype=bool)
            fuel_break_mask[sorted_indices[:num_breaks]] = True
            fuel_break_mask = fuel_break_mask.reshape(self.grid_size, self.grid_size)
            
            fuel_breaks[percentage] = fuel_break_mask
            
        return fuel_breaks
    
    def evaluate_fuel_break_performance(self, landscape_data: Dict[str, np.ndarray], 
                                      fuel_breaks: np.ndarray, 
                                      num_simulations: int = 10,
                                      max_duration: int = None) -> float:
        """
        Evaluate the performance of fuel breaks using fire simulation.
        
        Args:
            landscape_data: Dictionary containing landscape arrays
            fuel_breaks: Binary mask of fuel break locations
            num_simulations: Number of fire simulations to run
            
        Returns:
            Average acres burned (negative reward)
        """
        # Extract landscape arrays
        slope = landscape_data['slp']
        aspect = landscape_data['asp'] 
        dem = landscape_data['dem']
        cc = landscape_data['cc']
        cbd = landscape_data['cbd']
        cbh = landscape_data['cbh']
        ch = landscape_data['ch']
        fuel_model = landscape_data['fbfm']
        
        # Create simulator
        simulator = Simulate(slope, aspect, dem, cc, cbd, cbh, ch, fuel_model)
        simulator.set_space_time_cubes()
        
        # Apply fuel breaks
        simulator.set_fuel_breaks(fuel_breaks)
        
        # Run multiple simulations with optional max_duration
        simulator.run_many_simulations(num_simulations, max_duration)
        
        # Return negative acres burned as reward (minimize fire damage)
        acres_burned = simulator.get_loss()
        return -acres_burned
    
    def generate_training_memories(self, landscape_indices: List[int], 
                                 percentages: List[float] = None,
                                 num_simulations: int = 5,
                                 max_duration: int = None) -> List[Tuple]:
        """
        Generate training memories from domirank-based fuel breaks.
        
        Args:
            landscape_indices: List of landscape indices to process
            percentages: List of fuel break percentages to evaluate
            num_simulations: Number of simulations per evaluation
            
        Returns:
            List of (state, action, reward, next_state, done) tuples
        """
        if percentages is None:
            percentages = list(range(1, 21))  # 1% to 20%
        
        memories = []
        
        print(f"Generating memories for {len(landscape_indices)} landscapes...")
        
        for idx in landscape_indices:
            print(f"Processing landscape {idx}...")
            
            # Load landscape data
            landscape_data = self.load_landscape_data(idx)
            
            # Load domirank values
            domirank_values = self.load_domirank_values(idx)
            
            # Create iterative fuel breaks
            fuel_breaks_dict = self.create_iterative_fuel_breaks(domirank_values, percentages)
            
            # Convert landscape data to tensor format
            from src.scripts.DQNAgent import DQNAgent
            dummy_agent = DQNAgent()
            state_tensor = dummy_agent.preprocess_state(landscape_data)
            
            # Generate memories for incremental fuel break placement
            current_fuel_breaks = np.zeros((self.grid_size, self.grid_size), dtype=bool)
            
            for i, percentage in enumerate(percentages):
                target_fuel_breaks = fuel_breaks_dict[percentage]
                
                # Find new fuel breaks added at this step
                new_breaks = target_fuel_breaks & (~current_fuel_breaks)
                
                if np.any(new_breaks):
                    # Get action (location of new fuel break)
                    new_break_positions = np.where(new_breaks)
                    if len(new_break_positions[0]) > 0:
                        # Take the first new break as the action
                        row, col = new_break_positions[0][0], new_break_positions[1][0]
                        action = row * self.grid_size + col
                        
                        # Evaluate reward for this fuel break configuration
                        reward = self.evaluate_fuel_break_performance(
                            landscape_data, target_fuel_breaks, num_simulations, max_duration
                        )
                        
                        # Create experience tuple
                        # State: current landscape + current fuel breaks
                        next_state_tensor = state_tensor.clone()
                        
                        # Store memory
                        done = (i == len(percentages) - 1)  # Last step in sequence
                        
                        # Store states without batch dimension
                        state_to_store = state_tensor.squeeze(0) if state_tensor.dim() == 4 and state_tensor.size(0) == 1 else state_tensor
                        next_state_to_store = next_state_tensor.squeeze(0) if next_state_tensor.dim() == 4 and next_state_tensor.size(0) == 1 else next_state_tensor
                        
                        memories.append({
                            'state': state_to_store.cpu(),
                            'action': action,
                            'reward': reward,
                            'next_state': next_state_to_store.cpu(),
                            'done': done,
                            'landscape_idx': idx,
                            'percentage': percentage,
                            'fuel_breaks': current_fuel_breaks.copy()
                        })
                        
                        current_fuel_breaks = target_fuel_breaks.copy()
                        
                        print(f"  Added memory for {percentage}% fuel breaks, reward: {reward:.2f}")
        
        print(f"Generated {len(memories)} training memories")
        return memories
    
    def save_memories(self, memories: List[Dict], filepath: str):
        """Save memories to file."""
        torch.save(memories, filepath)
        print(f"Saved {len(memories)} memories to {filepath}")
    
    def load_memories(self, filepath: str) -> List[Dict]:
        """Load memories from file."""
        memories = torch.load(filepath)
        print(f"Loaded {len(memories)} memories from {filepath}")
        return memories
    
    def visualize_fuel_breaks(self, landscape_idx: int, percentages: List[float] = None):
        """Visualize fuel breaks at different percentages for a landscape."""
        if percentages is None:
            percentages = [1, 5, 10, 15, 20]
        
        # Load data
        landscape_data = self.load_landscape_data(landscape_idx)
        domirank_values = self.load_domirank_values(landscape_idx)
        
        # Create fuel breaks
        fuel_breaks_dict = self.create_iterative_fuel_breaks(domirank_values, percentages)
        
        # Create visualization
        fig, axes = plt.subplots(2, len(percentages), figsize=(20, 8))
        
        for i, percentage in enumerate(percentages):
            # Plot landscape (slope as example)
            axes[0, i].imshow(landscape_data['slp'], cmap='terrain')
            axes[0, i].set_title(f'Landscape {landscape_idx}\nSlope')
            axes[0, i].axis('off')
            
            # Plot fuel breaks
            fuel_breaks = fuel_breaks_dict[percentage]
            axes[1, i].imshow(landscape_data['slp'], cmap='terrain', alpha=0.7)
            axes[1, i].imshow(fuel_breaks, cmap='Reds', alpha=0.8)
            axes[1, i].set_title(f'{percentage}% Fuel Breaks')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'fuel_breaks_landscape_{landscape_idx}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Visualization saved as fuel_breaks_landscape_{landscape_idx}.png")