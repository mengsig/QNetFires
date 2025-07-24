#!/usr/bin/env python3
"""
Training monitoring script for QAgent.
Shows real-time metrics and helps understand training progress.
"""

import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import re

def parse_training_log(log_text):
    """Parse training log text and extract metrics."""
    metrics = {
        'episodes': [],
        'loss': [],
        'ep_reward': [],
        'step_reward': [],
        'burned_area': [],
        'epsilon': []
    }
    
    # Parse log lines
    lines = log_text.strip().split('\n')
    for line in lines:
        # Parse episode metrics
        if '[MetaEp' in line and 'loss=' in line:
            try:
                # Extract episode number
                ep_match = re.search(r'\[MetaEp (\d+)\]', line)
                if ep_match:
                    ep = int(ep_match.group(1))
                    metrics['episodes'].append(ep)
                
                # Extract metrics
                loss_match = re.search(r'loss=([\d\.-]+)', line)
                if loss_match:
                    metrics['loss'].append(float(loss_match.group(1)))
                else:
                    metrics['loss'].append(np.nan)
                
                ep_reward_match = re.search(r'ep_reward=([\d\.-]+)', line)
                if ep_reward_match:
                    metrics['ep_reward'].append(float(ep_reward_match.group(1)))
                else:
                    metrics['ep_reward'].append(np.nan)
                
                step_reward_match = re.search(r'step_reward=([\d\.-]+)', line)
                if step_reward_match:
                    metrics['step_reward'].append(float(step_reward_match.group(1)))
                else:
                    metrics['step_reward'].append(np.nan)
                
                burned_match = re.search(r'burned_area=([\d\.-]+)', line)
                if burned_match:
                    metrics['burned_area'].append(float(burned_match.group(1)))
                else:
                    metrics['burned_area'].append(np.nan)
                
                eps_match = re.search(r'eps=([\d\.-]+)', line)
                if eps_match:
                    metrics['epsilon'].append(float(eps_match.group(1)))
                else:
                    metrics['epsilon'].append(np.nan)
                    
            except (ValueError, AttributeError):
                continue
    
    return metrics

def create_monitoring_plot(metrics):
    """Create a comprehensive monitoring plot."""
    if not metrics['episodes']:
        print("No metrics to plot")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('QAgent Training Monitoring', fontsize=16, fontweight='bold')
    
    episodes = metrics['episodes']
    
    # Loss plot
    ax1 = axes[0, 0]
    if metrics['loss']:
        valid_loss = [(ep, loss) for ep, loss in zip(episodes, metrics['loss']) if not np.isnan(loss)]
        if valid_loss:
            ep_vals, loss_vals = zip(*valid_loss)
            ax1.plot(ep_vals, loss_vals, 'b-', linewidth=2, alpha=0.7)
            ax1.set_title('Training Loss')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Loss')
            ax1.grid(True, alpha=0.3)
    
    # Episode rewards
    ax2 = axes[0, 1]
    if metrics['ep_reward']:
        valid_ep_reward = [(ep, rew) for ep, rew in zip(episodes, metrics['ep_reward']) if not np.isnan(rew)]
        if valid_ep_reward:
            ep_vals, rew_vals = zip(*valid_ep_reward)
            ax2.plot(ep_vals, rew_vals, 'g-', linewidth=2, alpha=0.7)
            ax2.set_title('Episode Rewards')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Episode Return')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No completed episodes yet\n(This is normal early in training)', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Episode Rewards (Waiting for completion)')
    
    # Step rewards
    ax3 = axes[0, 2]
    if metrics['step_reward']:
        valid_step_reward = [(ep, rew) for ep, rew in zip(episodes, metrics['step_reward']) if not np.isnan(rew)]
        if valid_step_reward:
            ep_vals, rew_vals = zip(*valid_step_reward)
            ax3.plot(ep_vals, rew_vals, 'r-', linewidth=2, alpha=0.7)
            ax3.set_title('Step Rewards (Always Available)')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Average Step Reward')
            ax3.grid(True, alpha=0.3)
    
    # Burned area
    ax4 = axes[1, 0]
    if metrics['burned_area']:
        valid_burned = [(ep, burned) for ep, burned in zip(episodes, metrics['burned_area']) if not np.isnan(burned)]
        if valid_burned:
            ep_vals, burned_vals = zip(*valid_burned)
            ax4.plot(ep_vals, burned_vals, 'orange', linewidth=2, alpha=0.7)
            ax4.set_title('Burned Area (Lower is Better)')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Average Burned Area')
            ax4.grid(True, alpha=0.3)
    
    # Epsilon decay
    ax5 = axes[1, 1]
    if metrics['epsilon']:
        valid_eps = [(ep, eps) for ep, eps in zip(episodes, metrics['epsilon']) if not np.isnan(eps)]
        if valid_eps:
            ep_vals, eps_vals = zip(*valid_eps)
            ax5.plot(ep_vals, eps_vals, 'purple', linewidth=2, alpha=0.7)
            ax5.set_title('Exploration (Epsilon)')
            ax5.set_xlabel('Episode')
            ax5.set_ylabel('Epsilon')
            ax5.grid(True, alpha=0.3)
    
    # Summary statistics
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Calculate summary stats
    summary_text = "Training Summary:\n\n"
    
    if episodes:
        summary_text += f"Episodes completed: {len(episodes)}\n"
        
        if metrics['loss'] and any(not np.isnan(l) for l in metrics['loss']):
            recent_loss = [l for l in metrics['loss'][-10:] if not np.isnan(l)]
            if recent_loss:
                summary_text += f"Recent avg loss: {np.mean(recent_loss):.4f}\n"
        
        if metrics['step_reward'] and any(not np.isnan(r) for r in metrics['step_reward']):
            recent_step_reward = [r for r in metrics['step_reward'][-10:] if not np.isnan(r)]
            if recent_step_reward:
                summary_text += f"Recent step reward: {np.mean(recent_step_reward):.4f}\n"
        
        if metrics['ep_reward'] and any(not np.isnan(r) for r in metrics['ep_reward']):
            completed_episodes = [r for r in metrics['ep_reward'] if not np.isnan(r)]
            summary_text += f"Completed episodes: {len(completed_episodes)}\n"
            if completed_episodes:
                summary_text += f"Avg episode reward: {np.mean(completed_episodes):.3f}\n"
        else:
            summary_text += "No episodes completed yet\n(Episodes may be long)\n"
        
        if metrics['burned_area'] and any(not np.isnan(b) for b in metrics['burned_area']):
            recent_burned = [b for b in metrics['burned_area'][-10:] if not np.isnan(b)]
            if recent_burned:
                summary_text += f"Recent burned area: {np.mean(recent_burned):.1f}\n"
        
        if metrics['epsilon'] and any(not np.isnan(e) for e in metrics['epsilon']):
            current_eps = [e for e in metrics['epsilon'] if not np.isnan(e)]
            if current_eps:
                summary_text += f"Current epsilon: {current_eps[-1]:.3f}\n"
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=12, 
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    return fig

def monitor_training_file(log_file, update_interval=30):
    """Monitor training from log file."""
    print(f"Monitoring training from {log_file}")
    print(f"Update interval: {update_interval} seconds")
    print("Press Ctrl+C to stop monitoring")
    
    plt.ion()  # Interactive mode
    
    try:
        while True:
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f:
                        log_content = f.read()
                    
                    metrics = parse_training_log(log_content)
                    
                    if metrics['episodes']:
                        plt.clf()
                        fig = create_monitoring_plot(metrics)
                        plt.savefig('training_monitor.png', dpi=150, bbox_inches='tight')
                        plt.show()
                        plt.pause(0.1)
                        
                        # Print recent status
                        print(f"\n[{time.strftime('%H:%M:%S')}] Training Status:")
                        if metrics['episodes']:
                            print(f"  Latest episode: {metrics['episodes'][-1]}")
                            
                            if metrics['step_reward'] and not np.isnan(metrics['step_reward'][-1]):
                                print(f"  Step reward: {metrics['step_reward'][-1]:.4f}")
                            
                            if metrics['burned_area'] and not np.isnan(metrics['burned_area'][-1]):
                                print(f"  Burned area: {metrics['burned_area'][-1]:.1f}")
                            
                            completed_episodes = [r for r in metrics['ep_reward'] if not np.isnan(r)]
                            print(f"  Episodes completed: {len(completed_episodes)}")
                    else:
                        print(f"[{time.strftime('%H:%M:%S')}] Waiting for training data...")
                        
                except Exception as e:
                    print(f"Error reading log file: {e}")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] Waiting for log file: {log_file}")
            
            time.sleep(update_interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
        plt.ioff()

def main():
    """Main monitoring function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor QAgent training progress")
    parser.add_argument('--log_file', type=str, default='training.log', 
                       help='Path to training log file')
    parser.add_argument('--interval', type=int, default=30,
                       help='Update interval in seconds')
    parser.add_argument('--static', action='store_true',
                       help='Generate static plot from existing log')
    
    args = parser.parse_args()
    
    if args.static:
        if os.path.exists(args.log_file):
            with open(args.log_file, 'r') as f:
                log_content = f.read()
            
            metrics = parse_training_log(log_content)
            fig = create_monitoring_plot(metrics)
            plt.savefig('training_analysis.png', dpi=200, bbox_inches='tight')
            plt.show()
            print("Static analysis plot saved as 'training_analysis.png'")
        else:
            print(f"Log file not found: {args.log_file}")
    else:
        monitor_training_file(args.log_file, args.interval)

if __name__ == "__main__":
    main()