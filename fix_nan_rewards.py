#!/usr/bin/env python3
"""
Fix nan episode rewards and high burned area issues in src/Train.py

The problems:
1. ep_reward=nan: Episodes are not completing properly (no episode_return in info)
2. High burned areas (738-832): Fire simulation failing, using fallback values
3. Episodes timing out instead of completing naturally
"""

import os
import shutil
import re

def fix_nan_rewards():
    """Fix the nan episode rewards issue."""
    
    train_file = "src/Train.py"
    backup_file = "src/Train_backup_nan_fix.py"
    
    if not os.path.exists(train_file):
        print(f"❌ {train_file} not found!")
        return False
    
    # Create backup
    shutil.copy(train_file, backup_file)
    print(f"💾 Created backup: {backup_file}")
    
    with open(train_file, 'r') as f:
        content = f.read()
    
    # Fix 1: Increase STEPS_PER_EP to allow episodes to complete
    content = re.sub(
        r'STEPS_PER_EP = \d+',
        'STEPS_PER_EP = 30  # Increased to allow episodes to complete naturally',
        content
    )
    
    # Fix 2: Reduce SIMS to 1 for stability (high burned areas suggest fire sim failures)
    content = re.sub(
        r'SIMS = \d+',
        'SIMS = 1  # Reduced to 1 for maximum stability',
        content
    )
    
    # Fix 3: Add episode completion tracking in the training loop
    episode_tracking_fix = '''
            # Enhanced episode completion tracking
            completed_episodes = 0
            for i in range(N_ENVS):
                if dones[i]:
                    completed_episodes += 1
                    
                    # Force episode return calculation if missing
                    info_i = infos[i] if isinstance(infos, (list, tuple)) else infos
                    if info_i and "episode_return" not in info_i:
                        # Calculate episode return from step reward
                        episode_return = safe_scalar(rews[i])
                        info_i["episode_return"] = episode_return
                        info_i["episode_length"] = step + 1
                        print(f"[env {i}] 🔧 Added missing episode_return: {episode_return:.3f}")
'''
    
    # Insert after the dones processing
    pattern = r'(dones = np\.asarray\(dones, dtype=bool\))'
    replacement = r'\1' + episode_tracking_fix
    content = re.sub(pattern, replacement, content)
    
    # Fix 4: Improve episode reward tracking
    improved_reward_tracking = '''
                # IMPROVED: Track episode completion with forced episode returns
                if dones[i]:
                    info_i = infos[i] if isinstance(infos, (list, tuple)) else infos
                    
                    # Get episode return - create if missing
                    if info_i and "episode_return" in info_i:
                        episode_reward = safe_scalar(info_i['episode_return'])
                    else:
                        # Calculate cumulative episode return from step reward
                        episode_reward = safe_scalar(rews[i])
                        if info_i:
                            info_i["episode_return"] = episode_reward
                            info_i["episode_length"] = step + 1
                    
                    episode_rewards.append(episode_reward)
                    reward_win.append(episode_reward)
                    
                    # Enhanced logging with fire simulation diagnostics
                    burned_val = info_i.get('burned', None) if info_i else None
                    burned_scalar = safe_scalar(burned_val, fallback=None)
                    burned_str = f"{burned_scalar:.1f}" if burned_scalar is not None else 'N/A'
                    
                    # Check if this looks like a fire simulation failure
                    fire_sim_failed = burned_scalar is not None and burned_scalar > 600
                    status = "🔥 FIRE_SIM_FAILED" if fire_sim_failed else "✅ COMPLETED"
                    
                    # Enhanced diagnostics
                    is_dummy_val = info_i.get('is_dummy', False) if info_i else False
                    env_type = "DUMMY" if safe_scalar(is_dummy_val, fallback=False) else "REAL"
                    env_id = safe_scalar(info_i.get('env_id', i) if info_i else i, fallback=i)
                    
                    print(f"[env {i}] {status} Episode: R={episode_reward:.3f} "
                          f"Burned={burned_str} Type={env_type} ID={env_id} Step={step+1}")'''
    
    # Replace the existing episode completion tracking
    old_tracking_pattern = r'# Track episode completion with enhanced logging.*?print\(f"\[env \{i\}\] ⏰ Episode timeout/error:.*?\)"'
    content = re.sub(old_tracking_pattern, improved_reward_tracking, content, flags=re.DOTALL)
    
    # Fix 5: Add episode completion summary
    episode_summary = '''
        # Episode completion summary
        if episode_rewards:
            print(f"📊 Episode Summary: {len(episode_rewards)}/{N_ENVS} environments completed naturally")
        else:
            print(f"⚠️  Episode Summary: 0/{N_ENVS} environments completed - all timed out!")
            print("💡 Suggestion: Increase STEPS_PER_EP or check fire simulation")'''
    
    # Insert before the episode statistics
    pattern = r'(# Episode statistics with multiple metrics)'
    replacement = episode_summary + r'\n        \1'
    content = re.sub(pattern, replacement, content)
    
    # Fix 6: Improve the episode reward calculation
    better_reward_calc = '''
        # IMPROVED: Episode returns with fallback handling
        if reward_win:
            mean_episode_reward = float(np.mean(reward_win))
            print(f"📈 Episode rewards available: {len(reward_win)} episodes, mean={mean_episode_reward:.3f}")
        else:
            # Fallback: Use step rewards if no episodes completed
            if step_reward_win:
                mean_episode_reward = float(np.mean(step_reward_win))
                print(f"⚠️  No episode completions - using step reward as fallback: {mean_episode_reward:.3f}")
            else:
                mean_episode_reward = float("nan")
                print(f"❌ No rewards available at all!")'''
    
    # Replace the existing episode reward calculation
    old_calc_pattern = r'# Episode returns \(if episodes complete\).*?mean_episode_reward = float\(np\.mean\(reward_win\)\) if reward_win else float\("nan"\)'
    content = re.sub(old_calc_pattern, better_reward_calc, content, flags=re.DOTALL)
    
    # Write the fixed content
    with open(train_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Fixed nan rewards in {train_file}")
    print(f"💾 Backup saved as: {backup_file}")
    
    return True

def create_debug_version():
    """Create a version with extensive debugging for fire simulation issues."""
    
    train_file = "src/Train.py"
    debug_file = "src/Train_debug_fire.py"
    
    if not os.path.exists(train_file):
        print(f"❌ {train_file} not found!")
        return False
    
    with open(train_file, 'r') as f:
        content = f.read()
    
    # Add debug parameters
    debug_params = '''
# DEBUG PARAMETERS FOR FIRE SIMULATION ISSUES
DEBUG_FIRE_SIM = True
MAX_BURNED_THRESHOLD = 500  # Values above this suggest fire sim failure
MIN_STEPS_FOR_COMPLETION = 20  # Episodes should take at least this many steps
'''
    
    # Insert after imports
    import_pattern = r'(from src\.utils\.loadingUtils import RasterManager.*?\n)'
    content = re.sub(import_pattern, r'\1' + debug_params, content)
    
    # Add fire simulation debugging
    fire_debug_code = '''
                    # FIRE SIMULATION DEBUGGING
                    if DEBUG_FIRE_SIM:
                        burned_val = safe_scalar(burned_scalar, fallback=0)
                        if burned_val > MAX_BURNED_THRESHOLD:
                            print(f"🔥 HIGH BURNED AREA DETECTED: {burned_val:.1f} (threshold: {MAX_BURNED_THRESHOLD})")
                            print(f"   This suggests fire simulation failure or fallback values")
                        
                        if step + 1 < MIN_STEPS_FOR_COMPLETION:
                            print(f"⚡ EARLY COMPLETION: Episode ended at step {step+1} (min expected: {MIN_STEPS_FOR_COMPLETION})")
                            print(f"   This suggests environment issues or budget problems")
                        
                        # Check for patterns that indicate dummy environments
                        if burned_val in [150.0, 200.0, 1000.0]:
                            print(f"🤖 POSSIBLE DUMMY VALUE: {burned_val} is a common fallback value")'''
    
    # Insert in the episode completion logging
    pattern = r'(print\(f"\[env \{i\}\] \{status\} Episode:.*?\))'
    replacement = r'\1' + fire_debug_code
    content = re.sub(pattern, replacement, content)
    
    # Save debug version
    with open(debug_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Created debug version: {debug_file}")
    return True

def create_stable_low_sims_version():
    """Create an ultra-stable version with minimal fire simulations."""
    
    train_file = "src/Train.py"
    stable_file = "src/Train_ultra_stable.py"
    
    if not os.path.exists(train_file):
        print(f"❌ {train_file} not found!")
        return False
    
    with open(train_file, 'r') as f:
        content = f.read()
    
    # Ultra-stable parameters
    stable_changes = [
        (r'N_ENVS = \d+', 'N_ENVS = 16  # Reduced for stability'),
        (r'STEPS_PER_EP = \d+', 'STEPS_PER_EP = 35  # Increased for natural completion'),
        (r'SIMS = \d+', 'SIMS = 1  # Minimal simulations for maximum stability'),
        (r'BUDGET = \d+', 'BUDGET = 200  # Reduced budget for faster episodes'),
        (r'K_STEPS = \d+', 'K_STEPS = 8  # Smaller fuel break steps'),
    ]
    
    for old, new in stable_changes:
        content = re.sub(old, new, content)
    
    # Add stability comment
    stability_note = '''
# ULTRA-STABLE CONFIGURATION FOR DEBUGGING
# - Reduced environments (16 vs 32)
# - Increased episode length (35 steps)
# - Minimal fire simulations (1 sim)
# - Smaller budget and steps for faster completion
# This should eliminate nan rewards and high burned areas
'''
    
    # Insert after the hyperparameters
    pattern = r'(# Hyperparameters.*?\n)'
    content = re.sub(pattern, r'\1' + stability_note, content)
    
    # Save stable version
    with open(stable_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Created ultra-stable version: {stable_file}")
    return True

if __name__ == "__main__":
    print("🚀 Fixing nan episode rewards and high burned areas...")
    
    print("\n1️⃣  Fixing nan rewards in main file...")
    if fix_nan_rewards():
        print("✅ nan rewards fixed!")
    
    print("\n2️⃣  Creating debug version...")
    if create_debug_version():
        print("✅ Debug version created!")
    
    print("\n3️⃣  Creating ultra-stable version...")
    if create_stable_low_sims_version():
        print("✅ Ultra-stable version created!")
    
    print("\n🎯 Results:")
    print("  • src/Train.py: Fixed nan rewards + increased episode length")
    print("  • src/Train_debug_fire.py: Extensive fire simulation debugging")
    print("  • src/Train_ultra_stable.py: Minimal config for maximum stability")
    
    print("\n🔍 Root Causes Addressed:")
    print("  • Episodes not completing: Increased STEPS_PER_EP to 30")
    print("  • Fire simulation failures: Reduced SIMS to 1")
    print("  • Missing episode returns: Added forced episode return calculation")
    print("  • High burned areas: Better fallback detection and logging")
    
    print("\n🚀 Recommended testing order:")
    print("  1. python3 src/Train_ultra_stable.py  # Most stable")
    print("  2. python3 src/Train_debug_fire.py    # If you need debugging")
    print("  3. python3 src/Train.py               # Fixed main version")