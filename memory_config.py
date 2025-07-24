#!/usr/bin/env python3
"""
Memory-optimized configuration for QAgent training.
Use this for systems with limited GPU memory (< 4GB).
"""

# Memory-efficient hyperparameters
MEMORY_CONFIGS = {
    "low_memory": {
        "BATCH_SIZE": 16,
        "N_ENVS": 8,
        "BUFFER_CAP": 50_000,
        "USE_ENHANCED_MODEL": False,  # Use basic QNet
        "USE_PRIORITIZED_REPLAY": False,
        "USE_LR_SCHEDULER": True,
        "GRADIENT_ACCUMULATION_STEPS": 4,
        "TARGET_SYNC_EVERY": 500,
        "description": "For GPUs with < 2GB memory"
    },
    
    "medium_memory": {
        "BATCH_SIZE": 32,
        "N_ENVS": 16,
        "BUFFER_CAP": 75_000,
        "USE_ENHANCED_MODEL": True,  # Use memory-efficient enhanced model
        "USE_PRIORITIZED_REPLAY": True,
        "USE_LR_SCHEDULER": True,
        "GRADIENT_ACCUMULATION_STEPS": 2,
        "TARGET_SYNC_EVERY": 1000,
        "MEMORY_EFFICIENT": True,
        "description": "For GPUs with 2-4GB memory (recommended for your setup)"
    },
    
    "high_memory": {
        "BATCH_SIZE": 64,
        "N_ENVS": 32,
        "BUFFER_CAP": 100_000,
        "USE_ENHANCED_MODEL": True,  # Full enhanced model
        "USE_PRIORITIZED_REPLAY": True,
        "USE_LR_SCHEDULER": True,
        "GRADIENT_ACCUMULATION_STEPS": 1,
        "TARGET_SYNC_EVERY": 1000,
        "MEMORY_EFFICIENT": False,
        "description": "For GPUs with > 6GB memory"
    }
}

def get_config_for_gpu_memory(gpu_memory_gb):
    """
    Get recommended configuration based on GPU memory.
    
    Args:
        gpu_memory_gb: Available GPU memory in GB
        
    Returns:
        dict: Configuration dictionary
    """
    if gpu_memory_gb < 2:
        return MEMORY_CONFIGS["low_memory"]
    elif gpu_memory_gb < 6:
        return MEMORY_CONFIGS["medium_memory"]
    else:
        return MEMORY_CONFIGS["high_memory"]

def print_memory_recommendations():
    """Print memory optimization recommendations."""
    print("QAgent Memory Optimization Guide")
    print("=" * 50)
    print()
    
    for config_name, config in MEMORY_CONFIGS.items():
        print(f"{config_name.upper().replace('_', ' ')} CONFIGURATION:")
        print(f"  {config['description']}")
        print(f"  - Batch Size: {config['BATCH_SIZE']}")
        print(f"  - Environments: {config['N_ENVS']}")
        print(f"  - Enhanced Model: {config['USE_ENHANCED_MODEL']}")
        print(f"  - Prioritized Replay: {config['USE_PRIORITIZED_REPLAY']}")
        print()
    
    print("ADDITIONAL MEMORY SAVING TIPS:")
    print("1. Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    print("2. Close other GPU applications")
    print("3. Use mixed precision training (future enhancement)")
    print("4. Reduce SIMS (fire simulations) from 25 to 10-15")
    print()

if __name__ == "__main__":
    print_memory_recommendations()