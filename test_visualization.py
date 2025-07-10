#!/usr/bin/env python3
"""
Simple test to verify the visualization components work correctly.
"""

import os
import sys
import numpy as np

# Add src to path for imports
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '.'))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

print("🧪 Testing Visualization Components...")

try:
    # Test imports
    print("📦 Testing imports...")
    from src.scripts.DQNAgent import DQNAgent
    from src.scripts.DomiRankMemoryLoader import DomiRankMemoryLoader
    from src.scripts.FireEnv import FireEnv
    print("✅ All imports successful!")
    
    # Test landscape loading
    print("\n🏞️  Testing landscape loading...")
    memory_loader = DomiRankMemoryLoader(grid_size=25)
    
    try:
        landscape_data = memory_loader.load_landscape_data(0)
        print("✅ Landscape data loaded from existing files")
    except:
        print("📊 Generating sample landscape data...")
        memory_loader.generate_sample_data(1)
        landscape_data = memory_loader.load_landscape_data(0)
        print("✅ Sample landscape data generated and loaded")
    
    print(f"   - Landscape shape: {landscape_data['slp'].shape}")
    print(f"   - Available layers: {list(landscape_data.keys())}")
    
    # Test FireEnv
    print("\n🔥 Testing FireEnv...")
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
    
    fire_env.num_simulations = 2  # Quick test
    fire_env.max_duration = 10   # Quick test
    
    # Test environment step
    print("   - Testing environment step...")
    obs = fire_env.reset()
    print(f"   - Initial observation shape: {obs.shape}")
    
    # Create a simple test action
    test_action = np.zeros(fire_env.H * fire_env.W)
    test_action[100:110] = 1  # Add some fuel breaks
    
    obs, reward, done, info = fire_env.step(test_action)
    print(f"   - Step completed: reward={reward:.1f}, acres_burned={info['acres_burned']}")
    print("✅ FireEnv test successful!")
    
    # Test DQNAgent
    print("\n🧠 Testing DQNAgent...")
    agent = DQNAgent(
        input_channels=8,
        grid_size=25,
        batch_size=4
    )
    
    # Test agent preprocessing
    state_tensor = agent.preprocess_state(landscape_data)
    print(f"   - State tensor shape: {state_tensor.shape}")
    
    # Test agent action
    action = agent.act(state_tensor, np.zeros((25, 25), dtype=bool))
    print(f"   - Agent action: {action}")
    print("✅ DQNAgent test successful!")
    
    print("\n🎬 Visualization components are working correctly!")
    print("You can now use the visualization script:")
    print("   python launch_visualization.py")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    print("\nPlease check that all required files are in the correct locations:")
    print("- src/scripts/DQNAgent.py")
    print("- src/scripts/DomiRankMemoryLoader.py") 
    print("- src/scripts/FireEnv.py")
    print("- src/scripts/Simulate.py")
    print("- cropped_raster/ directory with landscape data")