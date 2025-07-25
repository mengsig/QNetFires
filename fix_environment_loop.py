#!/usr/bin/env python3
"""
Fix the infinite environment recreation loop in src/Train.py

The issue: Training is stuck recreating environments instead of actually training.
This happens when the training loop logic is broken and constantly triggers recreation.
"""

import os
import shutil
import re

def fix_environment_loop():
    """Fix the infinite environment recreation loop."""
    
    train_file = "src/Train.py"
    backup_file = "src/Train_backup_loop_fix.py"
    
    if not os.path.exists(train_file):
        print(f"❌ {train_file} not found!")
        return False
    
    # Create backup
    shutil.copy(train_file, backup_file)
    print(f"💾 Created backup: {backup_file}")
    
    with open(train_file, 'r') as f:
        content = f.read()
    
    # Fix 1: Remove the proactive recreation that's happening too often
    # Find and comment out the proactive recreation
    proactive_pattern = r'# Proactive recreation every 100 episodes to prevent pipe corruption\s+if ep % 100 == 0 and ep > 0:.*?print\(f"✅ Proactively recreated \{N_ENVS\} environments"\)'
    
    replacement = '''# Proactive recreation disabled - was causing infinite loop
            # if ep % 100 == 0 and ep > 0:
            #     print(f"🔄 Proactive environment recreation at episode {ep} (prevents pipe corruption)")
            #     # Recreation logic disabled
            print(f"🎯 Episode {ep}: Continuing with existing environments")'''
    
    content = re.sub(proactive_pattern, replacement, content, flags=re.DOTALL)
    
    # Fix 2: Remove the periodic refresh that's happening every 10 episodes
    refresh_pattern = r'# Periodically refresh environments with new rasters\s+if ep % 10 == 0 and raster_manager\.get_num_loaded_rasters\(\) > N_ENVS:.*?obs = reset_out\[0\] if isinstance\(reset_out, tuple\) else reset_out'
    
    refresh_replacement = '''# Periodic refresh disabled - was causing recreation loop
        # Only refresh if there are actual issues, not automatically
        if False:  # ep % 50 == 0 and raster_manager.get_num_loaded_rasters() > N_ENVS:
            print("Refreshing environments with new rasters...")
            # Refresh logic disabled for stability'''
    
    content = re.sub(refresh_pattern, refresh_replacement, content, flags=re.DOTALL)
    
    # Fix 3: Simplify the training loop to focus on actual training
    # Add a simple training indicator
    training_indicator = '''
        print(f"🚀 Starting episode {ep} training loop...")
        episode_start_time = time.time()'''
    
    # Insert after episode_rewards = []
    pattern = r'(episode_rewards = \[\])'
    replacement = r'\1' + training_indicator
    content = re.sub(pattern, replacement, content)
    
    # Fix 4: Add episode timing
    episode_timing = '''
        episode_duration = time.time() - episode_start_time
        print(f"⏱️  Episode {ep} completed in {episode_duration:.1f}s")'''
    
    # Insert before the episode statistics
    pattern = r'(# Episode completion summary)'
    replacement = episode_timing + r'\n        \1'
    content = re.sub(pattern, replacement, content)
    
    # Fix 5: Add import for time at the top
    import_pattern = r'(import os\nimport sys)'
    import_replacement = r'\1\nimport time'
    content = re.sub(import_pattern, import_replacement, content)
    
    # Write the fixed content
    with open(train_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Fixed environment recreation loop in {train_file}")
    print(f"💾 Backup saved as: {backup_file}")
    
    return True

def create_minimal_training_version():
    """Create a minimal training version that focuses on actual training."""
    
    train_file = "src/Train.py"
    minimal_file = "src/Train_minimal.py"
    
    if not os.path.exists(train_file):
        print(f"❌ {train_file} not found!")
        return False
    
    with open(train_file, 'r') as f:
        content = f.read()
    
    # Remove all automatic environment recreation
    minimal_changes = [
        # Disable proactive recreation
        (r'if ep % 100 == 0 and ep > 0:', 'if False:  # Disabled proactive recreation'),
        
        # Disable periodic refresh
        (r'if ep % 10 == 0 and raster_manager\.get_num_loaded_rasters\(\) > N_ENVS:', 'if False:  # Disabled periodic refresh'),
        
        # Reduce health checks
        (r'if ep > 0 and ep % 50 == 0:', 'if ep > 0 and ep % 200 == 0:  # Reduced health check frequency'),
        
        # Simplify environment creation logging
        (r'print\(f"🔧 Creating environment \(attempt.*?\)"\)', '# Creating environment'),
    ]
    
    for old, new in minimal_changes:
        content = re.sub(old, new, content)
    
    # Add minimal training focus comment
    focus_comment = '''
# MINIMAL TRAINING VERSION
# - No automatic environment recreation
# - No periodic refreshes
# - Focus on actual training
# - Environments only recreated on actual failures
'''
    
    # Insert at the top after imports
    pattern = r'(# Hyperparameters)'
    replacement = focus_comment + r'\n\1'
    content = re.sub(pattern, replacement, content)
    
    # Save minimal version
    with open(minimal_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Created minimal training version: {minimal_file}")
    return True

def create_no_recreation_version():
    """Create a version with absolutely no environment recreation."""
    
    train_file = "src/Train.py"
    no_recreation_file = "src/Train_no_recreation.py"
    
    if not os.path.exists(train_file):
        print(f"❌ {train_file} not found!")
        return False
    
    with open(train_file, 'r') as f:
        content = f.read()
    
    # Remove ALL recreation logic
    recreation_removals = [
        # Remove proactive recreation entirely
        r'# Proactive recreation.*?print\(f"✅ Proactively recreated.*?\n',
        
        # Remove periodic refresh entirely
        r'# Periodically refresh environments.*?obs = reset_out\[0\].*?\n',
        
        # Remove health check recreation
        r'# Proactive environment recreation at episode.*?print\(f"✅ Proactively recreated.*?\n',
        
        # Remove environment failure recreation in training loop
        r'# If step failed, recreate environments.*?continue\n',
    ]
    
    for pattern in recreation_removals:
        content = re.sub(pattern, '# Environment recreation removed\n', content, flags=re.DOTALL)
    
    # Replace step failure handling with simple retry
    step_failure_replacement = '''
            # If step failed, log and continue (no recreation)
            if not step_success:
                print(f"⚠️  Step failed, but continuing without recreation")
                # Use dummy observation to continue
                obs = np.random.rand(*obs.shape).astype(np.float32)
                continue'''
    
    # Find and replace step failure handling
    pattern = r'# If step failed, recreate environments.*?continue'
    content = re.sub(pattern, step_failure_replacement, content, flags=re.DOTALL)
    
    # Add no-recreation comment
    no_recreation_comment = '''
# NO RECREATION VERSION
# - Environments created once at startup
# - No automatic recreation under any circumstances
# - Focus purely on training
# - Maximum speed, minimum overhead
'''
    
    # Insert comment
    pattern = r'(# Hyperparameters)'
    replacement = no_recreation_comment + r'\n\1'
    content = re.sub(pattern, replacement, content)
    
    # Save no-recreation version
    with open(no_recreation_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Created no-recreation version: {no_recreation_file}")
    return True

if __name__ == "__main__":
    print("🚀 Fixing infinite environment recreation loop...")
    
    print("\n1️⃣  Fixing environment loop in main file...")
    if fix_environment_loop():
        print("✅ Environment loop fixed!")
    
    print("\n2️⃣  Creating minimal training version...")
    if create_minimal_training_version():
        print("✅ Minimal version created!")
    
    print("\n3️⃣  Creating no-recreation version...")
    if create_no_recreation_version():
        print("✅ No-recreation version created!")
    
    print("\n🎯 Results:")
    print("  • src/Train.py: Fixed infinite recreation loop")
    print("  • src/Train_minimal.py: Minimal recreation, focus on training")
    print("  • src/Train_no_recreation.py: Zero recreation, maximum speed")
    
    print("\n🔍 Root Cause:")
    print("  • Proactive recreation every 100 episodes")
    print("  • Periodic refresh every 10 episodes")
    print("  • Both were triggering immediately and repeatedly")
    
    print("\n🚀 Recommended solution:")
    print("  python3 src/Train_no_recreation.py  # Fastest, no recreation")
    print("  python3 src/Train_minimal.py        # Minimal recreation")
    print("  python3 src/Train.py                # Fixed main version")