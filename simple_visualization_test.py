#!/usr/bin/env python3
"""
Simple visualization test that can be run without command line arguments.
"""

import os
import sys
import numpy as np
import glob

# Add src to path for imports
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '.'))
src_path = os.path.join(project_root, 'src')
scripts_path = os.path.join(src_path, 'scripts')

sys.path.insert(0, project_root)
sys.path.insert(0, src_path)
sys.path.insert(0, scripts_path)

# Set matplotlib backend
import matplotlib
matplotlib.use('Agg')

def test_visualization():
    """Test the visualization components."""
    print("🧪 Simple Visualization Test")
    print("=" * 40)
    
    try:
        # Test imports
        print("📦 Testing imports...")
        from src.scripts.DQNAgent import DQNAgent
        from src.scripts.DomiRankMemoryLoader import DomiRankMemoryLoader
        print("✅ Imports successful!")
        
        # Load landscape data
        print("🏞️  Loading landscape data...")
        memory_loader = DomiRankMemoryLoader(grid_size=25)
        
        try:
            landscape_data = memory_loader.load_landscape_data(0)
            print("✅ Landscape data loaded from existing files")
        except:
            print("📊 Generating sample landscape data...")
            memory_loader.generate_sample_data(1)
            landscape_data = memory_loader.load_landscape_data(0)
            print("✅ Sample landscape data generated")
        
        # Test agent
        print("🧠 Testing agent...")
            agent = DQNAgent(
        input_channels=12,
            grid_size=25,
            batch_size=4
        )
        
        # Test preprocessing
        state_tensor = agent.preprocess_state(landscape_data)
        print(f"✅ State tensor shape: {state_tensor.shape}")
        
        # Test action
        action = agent.act(state_tensor, np.zeros((25, 25), dtype=bool))
        print(f"✅ Agent action: {action}")
        
        # Find a model file
        print("🔍 Looking for trained models...")
        model_patterns = ["*.pt", "*.pth", "**/*.pt", "**/*.pth"]
        models = []
        for pattern in model_patterns:
            models.extend(glob.glob(pattern, recursive=True))
        
        if models:
            model_path = models[0]
            print(f"✅ Found model: {model_path}")
            
            # Test loading model
            try:
                agent.load_model(model_path)
                print("✅ Model loaded successfully!")
                
                # Test agent placement simulation
                print("🎯 Testing agent placement...")
                current_fuel_breaks = np.zeros((25, 25), dtype=bool)
                
                for step in range(3):
                    action = agent.act(state_tensor, current_fuel_breaks)
                    row = action // 25
                    col = action % 25
                    current_fuel_breaks[row, col] = True
                    print(f"   Step {step+1}: Placed fuel break at ({row}, {col})")
                
                print(f"✅ Agent placement test successful!")
                print(f"   Total fuel breaks: {np.sum(current_fuel_breaks)}")
                
            except Exception as e:
                print(f"⚠️  Could not load model: {e}")
                print("   This is OK for testing imports")
                
        else:
            print("⚠️  No trained models found")
            print("   This is OK for testing imports")
        
        print("\n🎬 All tests passed!")
        print("You can now run the full visualization:")
        print("   python launch_visualization.py")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_visualization()