#!/usr/bin/env python3
"""
🎬 Easy Launcher for Fuel Break Visualization

This script makes it easy to visualize your trained DQN agent's fuel break placement
strategy without dealing with command-line arguments.
"""

import os
import sys
import glob
from pathlib import Path

def find_trained_models():
    """Find all trained model files in the project."""
    model_patterns = [
        "*.pt",
        "*.pth", 
        "models/*.pt",
        "models/*.pth",
        "checkpoints/*.pt",
        "checkpoints/*.pth",
        "src/models/*.pt",
        "src/models/*.pth",
        "**/*.pt",
        "**/*.pth"
    ]
    
    found_models = []
    for pattern in model_patterns:
        found_models.extend(glob.glob(pattern, recursive=True))
    
    # Remove duplicates and sort
    found_models = sorted(list(set(found_models)))
    
    return found_models

def main():
    """Interactive launcher for fuel break visualization."""
    print("🎬 Fuel Break Visualization Launcher")
    print("=" * 50)
    
    # Test imports first
    print("🔍 Testing imports...")
    try:
        os.system("python test_imports.py")
    except:
        print("⚠️  Could not run import test, proceeding anyway...")
    
    # Find trained models
    models = find_trained_models()
    
    if not models:
        print("❌ No trained models found!")
        print("   Please make sure you have trained a model first:")
        print("   python src/scripts/train_dqn_fuel_breaks_parallel.py")
        return
    
    print(f"📂 Found {len(models)} trained model(s):")
    for i, model in enumerate(models):
        file_size = os.path.getsize(model) / (1024 * 1024)  # MB
        print(f"   {i+1}. {model} ({file_size:.1f} MB)")
    
    # Model selection
    if len(models) == 1:
        selected_model = models[0]
        print(f"\n✅ Using model: {selected_model}")
    else:
        while True:
            try:
                choice = input(f"\n🎯 Select model (1-{len(models)}): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(models):
                    selected_model = models[idx]
                    break
                else:
                    print(f"❌ Please enter a number between 1 and {len(models)}")
            except ValueError:
                print("❌ Please enter a valid number")
    
    # Landscape selection
    print(f"\n🏞️  Landscape Selection:")
    print("   Available landscapes: 0, 1, 2, 3, 4, ...")
    
    while True:
        try:
            landscape = input("🎯 Select landscape (default: 0): ").strip()
            if landscape == "":
                landscape = 0
            else:
                landscape = int(landscape)
            break
        except ValueError:
            print("❌ Please enter a valid number")
    
    # Visualization options
    print(f"\n🎨 Visualization Options:")
    print("   1. Quick view (static plots only)")
    print("   2. Full visualization (static + animation)")
    print("   3. With comparison to domirank")
    print("   4. Save results to files")
    
    while True:
        try:
            vis_choice = input("🎯 Select option (1-4, default: 2): ").strip()
            if vis_choice == "":
                vis_choice = 2
            else:
                vis_choice = int(vis_choice)
            if 1 <= vis_choice <= 4:
                break
            else:
                print("❌ Please enter a number between 1 and 4")
        except ValueError:
            print("❌ Please enter a valid number")
    
    # Build command
    cmd_parts = [
        "python", "visualize_agent_fuel_breaks.py",
        "--model", f'"{selected_model}"',
        "--landscape", str(landscape)
    ]
    
    if vis_choice == 1:  # Quick view
        cmd_parts.append("--no_animation")
    elif vis_choice == 3:  # With comparison
        cmd_parts.append("--comparison")
    elif vis_choice == 4:  # Save results
        model_name = Path(selected_model).stem
        cmd_parts.extend([
            "--save_static", f"fuel_breaks_static_{model_name}_landscape_{landscape}.png",
            "--save_animation", f"fuel_breaks_animation_{model_name}_landscape_{landscape}.gif",
            "--report", f"validation_report_{model_name}_landscape_{landscape}.md"
        ])
    
    # Execute command
    cmd = " ".join(cmd_parts)
    print(f"\n🚀 Running visualization...")
    print(f"Command: {cmd}")
    print("=" * 50)
    
    os.system(cmd)
    
    print("\n" + "=" * 50)
    print("✅ Visualization completed!")
    
    if vis_choice == 4:
        print("📄 Files saved:")
        print(f"   - Static plot: fuel_breaks_static_{Path(selected_model).stem}_landscape_{landscape}.png")
        print(f"   - Animation: fuel_breaks_animation_{Path(selected_model).stem}_landscape_{landscape}.gif")
        print(f"   - Report: validation_report_{Path(selected_model).stem}_landscape_{landscape}.md")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Visualization cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check that all required files are in place and try again.")