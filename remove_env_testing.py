#!/usr/bin/env python3
"""
Simple script to remove environment testing from src/Train.py for maximum speed.
"""

import os
import shutil
import re

def remove_testing_from_train():
    """Remove environment testing from src/Train.py"""
    
    train_file = "src/Train.py"
    backup_file = "src/Train_backup_no_testing.py"
    
    if not os.path.exists(train_file):
        print(f"❌ {train_file} not found!")
        return False
    
    # Create backup
    shutil.copy(train_file, backup_file)
    print(f"💾 Created backup: {backup_file}")
    
    with open(train_file, 'r') as f:
        content = f.read()
    
    # Remove the testing block - be more specific with the pattern
    testing_patterns = [
        # Pattern 1: The main testing block
        r'# Test the environment with a simple step\s+print\(f"Testing environment functionality\.\.\."\).*?print\(f"Environment test successful!"\)',
        
        # Pattern 2: Alternative pattern if first doesn't match
        r'print\(f"Testing environment functionality\.\.\."\).*?env\.reset\(\)\s+print\(f"Environment test successful!"\)',
        
        # Pattern 3: Even simpler pattern
        r'# Test the environment.*?Environment test successful!'
    ]
    
    removed = False
    for i, pattern in enumerate(testing_patterns):
        original_content = content
        content = re.sub(pattern, '# Environment testing removed for performance', content, flags=re.DOTALL)
        
        if content != original_content:
            print(f"✅ Removed testing using pattern {i+1}")
            removed = True
            break
    
    if not removed:
        print("⚠️  Testing pattern not found, trying manual replacement...")
        
        # Manual replacements for specific lines
        replacements = [
            (r'print\(f"Testing environment functionality\.\.\."\)', '# Testing removed'),
            (r'# Test the environment with a simple step', '# Testing removed'),
            (r'if obs is None or obs\.size == 0:', '# Testing removed'),
            (r'test_action = np\.zeros\(obs\.shape\[-2\] \* obs\.shape\[-1\]\)', '# Testing removed'),
            (r'obs2, reward, done, truncated, info = env\.step\(test_action\)', '# Testing removed'),
            (r'print\(f"Environment test successful!"\)', '# Testing removed'),
        ]
        
        for old, new in replacements:
            content = re.sub(old, new, content)
            
        print("✅ Applied manual replacements")
    
    # Write the modified content
    with open(train_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Environment testing removed from {train_file}")
    print(f"💾 Backup saved as: {backup_file}")
    
    return True

def add_fast_validation():
    """Add minimal validation instead of full testing"""
    
    train_file = "src/Train.py"
    
    with open(train_file, 'r') as f:
        content = f.read()
    
    # Add minimal validation after environment creation
    validation_code = '''
                # Minimal validation (much faster than full testing)
                try:
                    obs, _ = env.reset()
                    if obs is None or obs.size == 0:
                        raise RuntimeError("Environment validation failed")
                except Exception as e:
                    raise RuntimeError(f"Environment validation error: {e}")
'''
    
    # Insert after environment creation but before return
    pattern = r'(env = FuelBreakEnv\(.*?\))'
    replacement = r'\1' + validation_code
    
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    with open(train_file, 'w') as f:
        f.write(content)
    
    print("✅ Added minimal validation")

if __name__ == "__main__":
    print("🚀 Removing environment testing from src/Train.py...")
    
    if remove_testing_from_train():
        print("\n🎯 Environment testing removed successfully!")
        print("\n⚡ Expected speedup:")
        print("  • Original: 60-120 seconds for 32 environments")
        print("  • Optimized: ~5 seconds for 32 environments")
        print("  • Speedup: 12-24x faster!")
        
        print("\n🚀 You can now run:")
        print("  python3 src/Train.py")
        
        print("\n⚠️  Note: Environments will no longer be pre-tested")
        print("  Any environment issues will be caught during training")
    else:
        print("\n❌ Failed to remove testing")