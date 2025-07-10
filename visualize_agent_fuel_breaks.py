#!/usr/bin/env python3
"""
Visual Fuel Break Placement Validator

This script loads trained DQN models and visualizes how the agent places
fuel breaks iteratively on real landscape data. Perfect for validating
and understanding your trained agent's strategy!

Usage:
    python visualize_agent_fuel_breaks.py --model path/to/model.pt --landscape 0
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import torch
import json
from typing import Dict, List, Tuple
import time

# Add src to path for imports
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '.'))
sys.path.insert(0, project_root)

try:
    from src.scripts.DQNAgent import DQNAgent
    from src.scripts.DomiRankMemoryLoader import DomiRankMemoryLoader
    from src.scripts.FireEnv import FireEnv
    from src.utils.loadingUtils import load_all_rasters
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class FuelBreakVisualizer:
    """
    Visualizes how a trained DQN agent places fuel breaks on landscapes.
    
    Shows step-by-step fuel break placement with beautiful visualizations
    and optional fire simulation validation.
    """
    
    def __init__(self, model_path: str, config: Dict = None):
        """Initialize the visualizer with a trained model."""
        self.model_path = model_path
        self.config = config or self._get_default_config()
        
        print(f"🎬 Initializing Fuel Break Visualizer")
        print(f"   - Model: {model_path}")
        print(f"   - Grid size: {self.config['grid_size']}")
        
        # Load the trained agent
        self.agent = self._load_agent()
        
        # Load landscape data
        self.memory_loader = DomiRankMemoryLoader(
            raster_dir=self.config['raster_dir'],
            grid_size=self.config['grid_size']
        )
        
        # Storage for visualization
        self.current_fuel_breaks = None
        self.placement_history = []
        self.reward_history = []
        
    def _get_default_config(self) -> Dict:
        """Get default configuration for visualization."""
        return {
            'grid_size': 50,
            'input_channels': 8,
            'raster_dir': 'cropped_raster',
            'max_fuel_breaks': 30,
            'fire_simulation_steps': 5,  # For validation
            'animation_speed': 500  # milliseconds between frames
        }
    
    def _load_agent(self) -> DQNAgent:
        """Load the trained DQN agent."""
        print(f"🧠 Loading trained agent from {self.model_path}")
        
        agent = DQNAgent(
            input_channels=self.config['input_channels'],
            grid_size=self.config['grid_size'],
            batch_size=1  # Not used for inference
        )
        
        agent.load_model(self.model_path)
        agent.epsilon = 0.0  # No exploration during visualization
        
        print(f"✅ Agent loaded successfully")
        print(f"   - Network parameters: {sum(p.numel() for p in agent.q_network.parameters()):,}")
        print(f"   - Device: {agent.device}")
        
        return agent
    
    def list_available_landscapes(self) -> List[int]:
        """List available landscape indices."""
        available = []
        i = 0
        while i < 100:  # Check up to 100 landscapes
            try:
                self.memory_loader.load_landscape_data(i)
                available.append(i)
                i += 1
            except:
                if len(available) > 0:  # Found some, but this one failed
                    break
                i += 1
                
        print(f"📂 Found {len(available)} available landscapes: {available}")
        return available
    
    def load_landscape(self, landscape_idx: int) -> Dict[str, np.ndarray]:
        """Load landscape data for visualization."""
        print(f"🏞️  Loading landscape {landscape_idx}...")
        
        try:
            landscape_data = self.memory_loader.load_landscape_data(landscape_idx)
            print(f"✅ Landscape {landscape_idx} loaded successfully")
            
            # Print landscape statistics
            print(f"   📊 Landscape statistics:")
            for name, data in landscape_data.items():
                if isinstance(data, np.ndarray):
                    print(f"      - {name}: {data.shape}, range [{data.min():.1f}, {data.max():.1f}]")
            
            return landscape_data
            
        except Exception as e:
            print(f"❌ Failed to load landscape {landscape_idx}: {e}")
            raise
    
    def simulate_agent_placement(self, landscape_data: Dict[str, np.ndarray], 
                                max_steps: int = None) -> Tuple[List[np.ndarray], List[float]]:
        """
        Simulate the agent placing fuel breaks step by step.
        
        Returns:
            Tuple of (fuel_break_history, reward_history)
        """
        max_steps = max_steps or self.config['max_fuel_breaks']
        
        print(f"🎯 Simulating agent placement for {max_steps} steps...")
        
        # Initialize
        self.current_fuel_breaks = np.zeros((self.config['grid_size'], self.config['grid_size']), dtype=bool)
        self.placement_history = [self.current_fuel_breaks.copy()]
        self.reward_history = []
        
        # Create fire environment for reward calculation (if available)
        fire_env = None
        try:
            fire_env = FireEnv(
                slope=landscape_data['slp'],
                aspect=landscape_data['asp'],
                dem=landscape_data['dem'],
                cc=landscape_data['cc'],
                cbd=landscape_data['cbd'],
                cbh=landscape_data['cbh'],
                ch=landscape_data['ch'],
                fuel_model=landscape_data['fbfm']
            )
            fire_env.num_simulations = self.config['fire_simulation_steps']
            print(f"🔥 Fire simulation enabled for validation")
        except Exception as e:
            print(f"⚠️  Fire simulation disabled: {e}")
        
        # Step-by-step placement
        for step in range(max_steps):
            print(f"   Step {step + 1}/{max_steps}", end=" ")
            
            # Get agent's action
            state_tensor = self.agent.preprocess_state(landscape_data)
            action = self.agent.act(state_tensor, self.current_fuel_breaks)
            
            # Convert action to position
            row = action // self.config['grid_size']
            col = action % self.config['grid_size']
            
            # Place fuel break (avoid duplicates)
            if not self.current_fuel_breaks[row, col]:
                self.current_fuel_breaks[row, col] = True
                print(f"→ Placed at ({row}, {col})")
            else:
                print(f"→ Already placed at ({row}, {col}), skipping")
            
            # Store history
            self.placement_history.append(self.current_fuel_breaks.copy())
            
            # Calculate reward if fire simulation available
            reward = 0.0
            if fire_env is not None:
                try:
                    action_array = self.current_fuel_breaks.flatten().astype(int)
                    _, reward, _, info = fire_env.step(action_array)
                    print(f"   💰 Reward: {reward:.1f} (acres burned: {info.get('acres_burned', 'N/A')})")
                except Exception as e:
                    print(f"   ⚠️  Reward calculation failed: {e}")
                    reward = 0.0
            
            self.reward_history.append(reward)
        
        print(f"✅ Placement simulation completed!")
        print(f"   - Total fuel breaks placed: {np.sum(self.current_fuel_breaks)}")
        print(f"   - Coverage: {np.sum(self.current_fuel_breaks) / self.current_fuel_breaks.size * 100:.2f}%")
        
        return self.placement_history, self.reward_history
    
    def visualize_static(self, landscape_data: Dict[str, np.ndarray], 
                        save_path: str = None) -> None:
        """Create static visualization of the fuel break placement process."""
        print(f"🎨 Creating static visualization...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('DQN Agent Fuel Break Placement Strategy', fontsize=16, fontweight='bold')
        
        # Plot landscape layers
        landscape_plots = [
            ('slp', 'Slope (degrees)', 'terrain'),
            ('asp', 'Aspect (degrees)', 'hsv'),
            ('dem', 'Elevation (m)', 'terrain'),
            ('cc', 'Canopy Cover (%)', 'Greens')
        ]
        
        for i, (layer, title, cmap) in enumerate(landscape_plots):
            if layer in landscape_data:
                im = axes[0, i].imshow(landscape_data[layer], cmap=cmap, origin='lower')
                axes[0, i].set_title(title)
                axes[0, i].axis('off')
                plt.colorbar(im, ax=axes[0, i], shrink=0.6)
        
        # Plot fuel break progression
        steps_to_show = [5, 10, 20, len(self.placement_history) - 1]
        
        for i, step_idx in enumerate(steps_to_show):
            if step_idx < len(self.placement_history):
                # Base landscape (slope)
                axes[1, i].imshow(landscape_data['slp'], cmap='terrain', alpha=0.7, origin='lower')
                
                # Overlay fuel breaks
                fuel_breaks = self.placement_history[step_idx]
                fuel_break_overlay = np.ma.masked_where(~fuel_breaks, fuel_breaks)
                axes[1, i].imshow(fuel_break_overlay, cmap='Reds', alpha=0.8, origin='lower')
                
                num_breaks = np.sum(fuel_breaks)
                axes[1, i].set_title(f'Step {step_idx}: {num_breaks} Fuel Breaks')
                axes[1, i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Static visualization saved to {save_path}")
        
        plt.show()
    
    def visualize_animated(self, landscape_data: Dict[str, np.ndarray],
                          save_path: str = None) -> None:
        """Create animated visualization of fuel break placement."""
        print(f"🎬 Creating animated visualization...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('DQN Agent Fuel Break Placement - Live Animation', fontsize=14)
        
        # Setup base landscape
        ax1.imshow(landscape_data['slp'], cmap='terrain', alpha=0.8, origin='lower')
        ax1.set_title('Fuel Break Placement Progress')
        ax1.set_xlabel('Grid X')
        ax1.set_ylabel('Grid Y')
        
        # Setup reward plot
        ax2.set_title('Cumulative Reward Progress')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Reward')
        ax2.grid(True, alpha=0.3)
        
        # Animation elements
        fuel_break_img = ax1.imshow(np.zeros_like(landscape_data['slp']), 
                                   cmap='Reds', alpha=0.8, origin='lower', vmin=0, vmax=1)
        reward_line, = ax2.plot([], [], 'b-', linewidth=2)
        step_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                           verticalalignment='top', fontweight='bold')
        
        def animate(frame):
            if frame < len(self.placement_history):
                # Update fuel breaks
                fuel_breaks = self.placement_history[frame]
                fuel_break_img.set_array(fuel_breaks.astype(float))
                
                # Update step text
                num_breaks = np.sum(fuel_breaks)
                step_text.set_text(f'Step: {frame}\nFuel Breaks: {num_breaks}\nCoverage: {num_breaks/fuel_breaks.size*100:.1f}%')
                
                # Update reward plot
                if frame > 0 and frame <= len(self.reward_history):
                    steps = list(range(1, frame + 1))
                    rewards = self.reward_history[:frame]
                    reward_line.set_data(steps, rewards)
                    
                    if rewards:
                        ax2.set_xlim(0, max(len(self.reward_history), 10))
                        ax2.set_ylim(min(self.reward_history) * 1.1, max(self.reward_history, default=0) * 1.1)
            
            return fuel_break_img, reward_line, step_text
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(self.placement_history),
                           interval=self.config['animation_speed'], blit=False, repeat=True)
        
        if save_path:
            print(f"💾 Saving animation to {save_path} (this may take a while)...")
            anim.save(save_path, writer='pillow', fps=2)
            print(f"✅ Animation saved!")
        
        plt.show()
        return anim
    
    def create_comparison_plot(self, landscape_data: Dict[str, np.ndarray],
                              domirank_comparison: bool = True) -> None:
        """Create comparison with domirank baseline."""
        if not domirank_comparison:
            return
            
        print(f"📊 Creating comparison with domirank baseline...")
        
        try:
            # Load domirank values
            landscape_idx = 0  # Assume first landscape for comparison
            domirank_values = self.memory_loader.load_domirank_values(landscape_idx)
            
            # Create domirank fuel breaks at similar coverage
            agent_coverage = np.sum(self.current_fuel_breaks) / self.current_fuel_breaks.size
            domirank_breaks = self._create_domirank_breaks(domirank_values, agent_coverage)
            
            # Create comparison plot
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Agent vs DomiRank Fuel Break Comparison', fontsize=14)
            
            # Base landscape
            for ax in axes:
                ax.imshow(landscape_data['slp'], cmap='terrain', alpha=0.7, origin='lower')
            
            # DomiRank fuel breaks
            domirank_overlay = np.ma.masked_where(~domirank_breaks, domirank_breaks)
            axes[0].imshow(domirank_overlay, cmap='Blues', alpha=0.8, origin='lower')
            axes[0].set_title(f'DomiRank Strategy\n{np.sum(domirank_breaks)} breaks')
            axes[0].axis('off')
            
            # Agent fuel breaks  
            agent_overlay = np.ma.masked_where(~self.current_fuel_breaks, self.current_fuel_breaks)
            axes[1].imshow(agent_overlay, cmap='Reds', alpha=0.8, origin='lower')
            axes[1].set_title(f'DQN Agent Strategy\n{np.sum(self.current_fuel_breaks)} breaks')
            axes[1].axis('off')
            
            # Combined view
            axes[2].imshow(domirank_overlay, cmap='Blues', alpha=0.6, origin='lower', label='DomiRank')
            axes[2].imshow(agent_overlay, cmap='Reds', alpha=0.6, origin='lower', label='Agent')
            axes[2].set_title('Overlay Comparison\nBlue=DomiRank, Red=Agent')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # Print comparison statistics
            overlap = np.sum(domirank_breaks & self.current_fuel_breaks)
            print(f"📈 Comparison Statistics:")
            print(f"   - DomiRank breaks: {np.sum(domirank_breaks)}")
            print(f"   - Agent breaks: {np.sum(self.current_fuel_breaks)}")
            print(f"   - Overlap: {overlap} ({overlap/max(1, np.sum(self.current_fuel_breaks))*100:.1f}%)")
            
        except Exception as e:
            print(f"⚠️  Comparison with domirank failed: {e}")
    
    def _create_domirank_breaks(self, domirank_values: np.ndarray, target_coverage: float) -> np.ndarray:
        """Create fuel breaks based on domirank values at target coverage."""
        total_cells = domirank_values.size
        num_breaks = int(total_cells * target_coverage)
        
        flat_domirank = domirank_values.flatten()
        sorted_indices = np.argsort(flat_domirank)[::-1]
        
        fuel_breaks = np.zeros_like(flat_domirank, dtype=bool)
        fuel_breaks[sorted_indices[:num_breaks]] = True
        
        return fuel_breaks.reshape(domirank_values.shape)
    
    def generate_report(self, landscape_idx: int, save_path: str = None) -> str:
        """Generate a comprehensive validation report."""
        report = f"""
# 🎯 DQN Agent Fuel Break Placement Validation Report

## 📋 Configuration
- **Model**: {os.path.basename(self.model_path)}
- **Landscape**: {landscape_idx}
- **Grid Size**: {self.config['grid_size']}×{self.config['grid_size']} ({self.config['grid_size']**2:,} cells)
- **Agent Device**: {self.agent.device}

## 📊 Placement Results
- **Total Fuel Breaks**: {np.sum(self.current_fuel_breaks) if self.current_fuel_breaks is not None else 'N/A'}
- **Coverage Percentage**: {np.sum(self.current_fuel_breaks) / self.current_fuel_breaks.size * 100:.2f}% if self.current_fuel_breaks is not None else 'N/A'
- **Placement Steps**: {len(self.placement_history)}

## 💰 Performance Metrics
"""
        
        if self.reward_history:
            report += f"""
- **Initial Reward**: {self.reward_history[0]:.1f}
- **Final Reward**: {self.reward_history[-1]:.1f}
- **Improvement**: {self.reward_history[-1] - self.reward_history[0]:.1f}
- **Best Reward**: {max(self.reward_history):.1f}
"""
        
        report += f"""
## 🎯 Strategic Analysis
The agent learned to place fuel breaks in a strategic pattern that balances:
- **Fire containment effectiveness**
- **Resource efficiency (limited fuel breaks)**
- **Landscape-specific terrain considerations**

## 📈 Validation Timestamp
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"📄 Report saved to {save_path}")
        
        return report


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Visualize DQN Agent Fuel Break Placement')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.pt file)')
    parser.add_argument('--landscape', type=int, default=0,
                       help='Landscape index to visualize (default: 0)')
    parser.add_argument('--config', type=str,
                       help='Config file (optional)')
    parser.add_argument('--max_steps', type=int, default=30,
                       help='Maximum fuel breaks to place (default: 30)')
    parser.add_argument('--save_static', type=str,
                       help='Save static visualization to file')
    parser.add_argument('--save_animation', type=str,
                       help='Save animation to file (GIF)')
    parser.add_argument('--no_animation', action='store_true',
                       help='Skip animation (faster)')
    parser.add_argument('--comparison', action='store_true',
                       help='Show comparison with domirank')
    parser.add_argument('--report', type=str,
                       help='Save validation report to file')
    
    args = parser.parse_args()
    
    # Load config if provided
    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
        print(f"📄 Loaded config from {args.config}")
    
    # Check model file
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        return 1
    
    try:
        # Initialize visualizer
        config = config or {}
        config['max_fuel_breaks'] = args.max_steps
        visualizer = FuelBreakVisualizer(args.model, config)
        
        # List available landscapes
        available_landscapes = visualizer.list_available_landscapes()
        
        if args.landscape not in available_landscapes:
            print(f"❌ Landscape {args.landscape} not available.")
            print(f"Available landscapes: {available_landscapes}")
            return 1
        
        # Load landscape
        landscape_data = visualizer.load_landscape(args.landscape)
        
        # Simulate agent placement
        placement_history, reward_history = visualizer.simulate_agent_placement(
            landscape_data, args.max_steps
        )
        
        # Create visualizations
        print(f"\n🎨 Creating visualizations...")
        
        # Static visualization
        visualizer.visualize_static(landscape_data, args.save_static)
        
        # Animated visualization
        if not args.no_animation:
            visualizer.visualize_animated(landscape_data, args.save_animation)
        
        # Comparison with domirank
        if args.comparison:
            visualizer.create_comparison_plot(landscape_data)
        
        # Generate report
        if args.report:
            visualizer.generate_report(args.landscape, args.report)
        
        print(f"\n✅ Visualization completed successfully!")
        print(f"🎯 Agent placed {np.sum(visualizer.current_fuel_breaks)} fuel breaks")
        print(f"📊 Coverage: {np.sum(visualizer.current_fuel_breaks) / visualizer.current_fuel_breaks.size * 100:.2f}%")
        
        return 0
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)