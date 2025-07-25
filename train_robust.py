#!/usr/bin/env python3
"""
Robust training script for QAgent that handles multiprocessing errors.
This version includes comprehensive error handling and recovery mechanisms.
"""

import os
import sys
import signal
import time
from contextlib import contextmanager

# Add src to path
sys.path.append('src')

# Timeout handler for hanging operations
class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Operation timed out")
    
    # Set the signal handler and a timeout alarm
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Restore the old signal handler and cancel the alarm
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)

def main():
    """Run robust training with error recovery."""
    print("QAgent Robust Training with Error Recovery")
    print("=" * 50)
    
    # Set environment variables for stability
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # For better error reporting
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            print(f"\nAttempt {retry_count + 1}/{max_retries}")
            
            # Import and run training with timeout
            with timeout(3600):  # 1 hour timeout
                from Train import main as train_main
                train_main()
                
            print("✅ Training completed successfully!")
            return 0
            
        except TimeoutException:
            print("⚠ Training timed out, restarting...")
            retry_count += 1
            
        except (EOFError, BrokenPipeError, ConnectionResetError) as e:
            print(f"⚠ Multiprocessing error: {e}")
            print("This is usually caused by environment crashes in subprocesses")
            retry_count += 1
            
            # Clean up any remaining processes
            cleanup_processes()
            
        except KeyboardInterrupt:
            print("\n🛑 Training interrupted by user")
            cleanup_processes()
            return 1
            
        except Exception as e:
            print(f"❌ Training failed with error: {e}")
            import traceback
            traceback.print_exc()
            retry_count += 1
            
        if retry_count < max_retries:
            print(f"Waiting 10 seconds before retry...")
            time.sleep(10)
    
    print(f"❌ Training failed after {max_retries} attempts")
    print("\nTroubleshooting suggestions:")
    print("1. Try reducing the number of environments (N_ENVS)")
    print("2. Try reducing fire simulations (SIMS)")
    print("3. Check if pyretechnics is properly installed")
    print("4. Try running with basic QNet model")
    
    return 1

def cleanup_processes():
    """Clean up any hanging multiprocessing processes."""
    print("Cleaning up processes...")
    
    try:
        import psutil
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
                
        # Wait for processes to terminate
        psutil.wait_procs(children, timeout=5)
        
        # Force kill any remaining processes
        for child in children:
            try:
                if child.is_running():
                    child.kill()
            except psutil.NoSuchProcess:
                pass
                
    except ImportError:
        # If psutil not available, use basic cleanup
        import multiprocessing
        for p in multiprocessing.active_children():
            p.terminate()
            p.join(timeout=5)
            if p.is_alive():
                p.kill()

if __name__ == "__main__":
    exit(main())