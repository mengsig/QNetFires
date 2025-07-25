#!/usr/bin/env python3
"""
Memory-efficient training script for QAgent.
Optimized for systems with limited GPU memory (< 4GB).
"""

import os
import sys

# Add src to path
sys.path.append('src')

def main():
    """Run memory-efficient training."""
    print("QAgent Memory-Efficient Training")
    print("=" * 40)
    
    # Check GPU memory
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"Detected GPU memory: {gpu_memory:.2f} GB")
            
            if gpu_memory < 4:
                print("⚠ Limited GPU memory detected!")
                print("Using memory-optimized settings...")
                
                # Set environment variable for memory management
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
                
                # Override settings in Train.py by modifying the file temporarily
                modify_train_for_memory_efficiency()
            else:
                print("✓ Sufficient GPU memory available")
        else:
            print("Using CPU training")
    except ImportError:
        print("PyTorch not available")
        return 1
    
    # Import and run training
    try:
        from Train import main as train_main
        train_main()
    except Exception as e:
        print(f"Training failed: {e}")
        print("\nTroubleshooting tips:")
        print("1. Try reducing BATCH_SIZE further")
        print("2. Try reducing N_ENVS further") 
        print("3. Use basic QNet model instead of enhanced")
        print("4. Close other GPU applications")
        return 1
    
    return 0

def modify_train_for_memory_efficiency():
    """Temporarily modify Train.py for memory efficiency."""
    print("Applying memory optimizations...")
    
    # Read current Train.py
    with open('src/Train.py', 'r') as f:
        content = f.read()
    
    # Apply memory-efficient modifications
    modifications = [
        ('BATCH_SIZE = 32', 'BATCH_SIZE = 16'),
        ('N_ENVS = 16', 'N_ENVS = 8'),
        ('SIMS = 25', 'SIMS = 15'),
        ('BUFFER_CAP = 100_000', 'BUFFER_CAP = 50_000'),
        ('USE_ENHANCED_MODEL = True', 'USE_ENHANCED_MODEL = False'),
        ('USE_PRIORITIZED_REPLAY = True', 'USE_PRIORITIZED_REPLAY = False'),
    ]
    
    modified_content = content
    for old, new in modifications:
        if old in modified_content:
            modified_content = modified_content.replace(old, new)
            print(f"  Modified: {old} -> {new}")
    
    # Write temporary file
    with open('src/Train_memory.py', 'w') as f:
        f.write(modified_content)
    
    # Replace import
    sys.path.insert(0, 'src')
    import Train_memory as Train
    globals()['Train'] = Train

if __name__ == "__main__":
    exit(main())