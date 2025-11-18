#!/usr/bin/env python3
"""
Evaluation script for testing trained humanoid walking models.
This script loads a trained model and evaluates its walking performance.
"""

import torch
import numpy as np
import os
import argparse
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from HumanoidWalkEnv import HumanoidWalkEnv
import matplotlib.pyplot as plt
from collections import deque
import time


def evaluate_model(model_path, vec_normalize_path=None, num_episodes=10, render=True, 
                  xml_path="assets/humanoid_180_75.xml", verbose=True):
    """
    Evaluate a trained humanoid walking model.
    
    Args:
        model_path: Path to the trained model
        vec_normalize_path: Optional path to VecNormalize statistics
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        xml_path: Path to the humanoid XML file
        verbose: Whether to print detailed statistics
    
    Returns:
        Dictionary containing evaluation metrics
    """
    
    # Create environment
    env = HumanoidWalkEnv(xml_file=xml_path, training_phase="walking")
    env.set_training_phase("walking", progress=1.0)  # Full walking mode
    
    # Wrap in DummyVecEnv for compatibility
    vec_env = DummyVecEnv([lambda: env])
    
    # Load VecNormalize if provided
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        vec_env = VecNormalize.load(vec_normalize_path, vec_env)
        vec_env.training = False  # Set to evaluation mode
        vec_env.norm_reward = False
        print(f"✓ Loaded VecNormalize from {vec_normalize_path}")
    
    # Load the model
    model = TQC.load(model_path, env=vec_env, device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Loaded model from {model_path}")
    
    # Metrics storage
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'forward_distances': [],
        'average_velocities': [],
        'fall_count': 0,
        'max_distance': 0,
        'contact_patterns': [],
        'foot_clearances': [],
    }
    
    # Evaluation loop
    for episode in range(num_episodes):
        obs = vec_env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        episode_info = []
        
        # Track position
        start_x = env.data.qpos[0]
        max_x = start_x
        
        # Track gait metrics
        single_support_count = 0
        double_support_count = 0
        no_contact_count = 0
        max_clearance = 0
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"{'='*50}")
        
        while not done:
            # Get action from policy
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, done, info = vec_env.step(action)
            
            # Update metrics
            episode_reward += reward[0]
            episode_length += 1
            
            # Extract detailed info
            if info[0]:
                current_x = info[0].get('env_metrics/x_position', 0)
                max_x = max(max_x, current_x)
                
                # Track contact patterns
                if info[0].get('env_metrics/single_support', 0):
                    single_support_count += 1
                elif info[0].get('env_metrics/both_contact', 0):
                    double_support_count += 1
                elif info[0].get('env_metrics/no_contact', 0):
                    no_contact_count += 1
                
                # Track foot clearance
                clearance = info[0].get('gait_reward/clearance_achieved', 0)
                max_clearance = max(max_clearance, clearance)
            
            # Render if requested
            if render:
                vec_env.render("human")
                time.sleep(0.01)  # Slow down for visibility
            
            # Check for early termination (fall)
            if done and episode_length < 1000:
                metrics['fall_count'] += 1
                if verbose:
                    print(f"  ⚠️ Episode ended early (fall) at step {episode_length}")
        
        # Calculate episode statistics
        forward_distance = max_x - start_x
        avg_velocity = forward_distance / (episode_length * env.dt)
        
        # Store metrics
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_lengths'].append(episode_length)
        metrics['forward_distances'].append(forward_distance)
        metrics['average_velocities'].append(avg_velocity)
        metrics['max_distance'] = max(metrics['max_distance'], forward_distance)
        
        # Calculate gait quality
        total_steps = single_support_count + double_support_count + no_contact_count
        if total_steps > 0:
            single_support_ratio = single_support_count / total_steps
            double_support_ratio = double_support_count / total_steps
            flight_ratio = no_contact_count / total_steps
            
            metrics['contact_patterns'].append({
                'single_support': single_support_ratio,
                'double_support': double_support_ratio,
                'flight_phase': flight_ratio
            })
        
        metrics['foot_clearances'].append(max_clearance)
        
        # Print episode summary
        if verbose:
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Length: {episode_length} steps")
            print(f"  Distance: {forward_distance:.2f}m")
            print(f"  Avg Velocity: {avg_velocity:.2f}m/s")
            print(f"  Max Clearance: {max_clearance:.4f}m")
            if total_steps > 0:
                print(f"  Gait: Single={single_support_ratio:.1%}, "
                      f"Double={double_support_ratio:.1%}, "
                      f"Flight={flight_ratio:.1%}")
    
    # Calculate summary statistics
    summary = {
        'mean_reward': np.mean(metrics['episode_rewards']),
        'std_reward': np.std(metrics['episode_rewards']),
        'mean_distance': np.mean(metrics['forward_distances']),
        'std_distance': np.std(metrics['forward_distances']),
        'mean_velocity': np.mean(metrics['average_velocities']),
        'std_velocity': np.std(metrics['average_velocities']),
        'success_rate': 1.0 - (metrics['fall_count'] / num_episodes),
        'max_distance': metrics['max_distance'],
        'mean_clearance': np.mean(metrics['foot_clearances']),
    }
    
    # Calculate average gait pattern
    if metrics['contact_patterns']:
        avg_single = np.mean([p['single_support'] for p in metrics['contact_patterns']])
        avg_double = np.mean([p['double_support'] for p in metrics['contact_patterns']])
        avg_flight = np.mean([p['flight_phase'] for p in metrics['contact_patterns']])
        summary['avg_gait_pattern'] = {
            'single_support': avg_single,
            'double_support': avg_double,
            'flight_phase': avg_flight
        }
    
    # Print final summary
    print(f"\n{'='*60}")
    print(f"{'EVALUATION SUMMARY':^60}")
    print(f"{'='*60}")
    print(f"Episodes: {num_episodes}")
    print(f"Success Rate: {summary['success_rate']*100:.1f}%")
    print(f"Mean Reward: {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
    print(f"Mean Distance: {summary['mean_distance']:.2f}m ± {summary['std_distance']:.2f}m")
    print(f"Mean Velocity: {summary['mean_velocity']:.2f}m/s ± {summary['std_velocity']:.2f}m/s")
    print(f"Max Distance: {summary['max_distance']:.2f}m")
    print(f"Mean Foot Clearance: {summary['mean_clearance']*1000:.1f}mm")
    
    if 'avg_gait_pattern' in summary:
        print(f"\nGait Analysis:")
        print(f"  Single Support: {summary['avg_gait_pattern']['single_support']*100:.1f}%")
        print(f"  Double Support: {summary['avg_gait_pattern']['double_support']*100:.1f}%")
        print(f"  Flight Phase: {summary['avg_gait_pattern']['flight_phase']*100:.1f}%")
    
    # Determine walking quality
    print(f"\n{'Walking Quality Assessment':^60}")
    print("-" * 60)
    
    quality_score = 0
    quality_notes = []
    
    # Check success rate
    if summary['success_rate'] > 0.9:
        quality_score += 25
        quality_notes.append("✓ Excellent stability (90%+ success)")
    elif summary['success_rate'] > 0.7:
        quality_score += 15
        quality_notes.append("○ Good stability (70%+ success)")
    else:
        quality_score += 5
        quality_notes.append("✗ Poor stability (<70% success)")
    
    # Check velocity
    if 0.5 < summary['mean_velocity'] < 2.0:
        quality_score += 25
        quality_notes.append("✓ Natural walking speed")
    elif 0.2 < summary['mean_velocity'] < 3.0:
        quality_score += 15
        quality_notes.append("○ Acceptable speed")
    else:
        quality_score += 5
        quality_notes.append("✗ Unnatural speed")
    
    # Check gait pattern
    if 'avg_gait_pattern' in summary:
        if summary['avg_gait_pattern']['flight_phase'] < 0.05:
            quality_score += 25
            quality_notes.append("✓ No jumping/hopping detected")
        elif summary['avg_gait_pattern']['flight_phase'] < 0.15:
            quality_score += 15
            quality_notes.append("○ Minimal flight phase")
        else:
            quality_score += 5
            quality_notes.append("✗ Excessive jumping/hopping")
        
        if 0.3 < summary['avg_gait_pattern']['single_support'] < 0.7:
            quality_score += 25
            quality_notes.append("✓ Good alternating gait")
        else:
            quality_score += 10
            quality_notes.append("○ Suboptimal gait pattern")
    
    # Print assessment
    for note in quality_notes:
        print(f"  {note}")
    
    print(f"\nOverall Walking Quality Score: {quality_score}/100")
    
    if quality_score >= 80:
        print("🏆 EXCELLENT: Natural bipedal walking achieved!")
    elif quality_score >= 60:
        print("👍 GOOD: Functional walking with minor issues")
    elif quality_score >= 40:
        print("⚠️ FAIR: Walking present but needs improvement")
    else:
        print("❌ POOR: Significant walking issues detected")
    
    print(f"{'='*60}")
    
    return summary, metrics


def plot_training_curves(log_path, save_path=None):
    """
    Plot training curves from TensorBoard logs.
    
    Args:
        log_path: Path to TensorBoard log directory
        save_path: Optional path to save the plot
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        # Load TensorBoard logs
        event_acc = EventAccumulator(log_path)
        event_acc.Reload()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Humanoid Walking Training Progress', fontsize=16)
        
        # Plot reward
        if 'rollout/ep_rew_mean' in event_acc.Tags()['scalars']:
            rewards = event_acc.Scalars('rollout/ep_rew_mean')
            steps = [r.step for r in rewards]
            values = [r.value for r in rewards]
            axes[0, 0].plot(steps, values, label='Episode Reward')
            axes[0, 0].set_xlabel('Steps')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].set_title('Mean Episode Reward')
            axes[0, 0].grid(True)
        
        # Plot forward velocity
        if 'env_metrics/forward_velocity' in event_acc.Tags()['scalars']:
            velocity = event_acc.Scalars('env_metrics/forward_velocity')
            steps = [v.step for v in velocity]
            values = [v.value for v in velocity]
            axes[0, 1].plot(steps, values, label='Forward Velocity', color='green')
            axes[0, 1].set_xlabel('Steps')
            axes[0, 1].set_ylabel('Velocity (m/s)')
            axes[0, 1].set_title('Forward Velocity')
            axes[0, 1].grid(True)
        
        # Plot curriculum progress
        if 'curriculum/progress' in event_acc.Tags()['scalars']:
            progress = event_acc.Scalars('curriculum/progress')
            steps = [p.step for p in progress]
            values = [p.value for p in progress]
            axes[1, 0].plot(steps, values, label='Curriculum Progress', color='orange')
            axes[1, 0].set_xlabel('Steps')
            axes[1, 0].set_ylabel('Progress')
            axes[1, 0].set_title('Curriculum Progress')
            axes[1, 0].grid(True)
        
        # Plot gait metrics
        if 'env_metrics/single_support' in event_acc.Tags()['scalars']:
            single = event_acc.Scalars('env_metrics/single_support')
            steps = [s.step for s in single]
            values = [s.value for s in single]
            axes[1, 1].plot(steps, values, label='Single Support', color='blue')
        
        if 'env_metrics/no_contact' in event_acc.Tags()['scalars']:
            no_contact = event_acc.Scalars('env_metrics/no_contact')
            steps = [n.step for n in no_contact]
            values = [n.value for n in no_contact]
            axes[1, 1].plot(steps, values, label='Flight Phase', color='red')
        
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Ratio')
        axes[1, 1].set_title('Gait Pattern')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Plot saved to {save_path}")
        
        plt.show()
        
    except ImportError:
        print("TensorBoard not installed. Install with: pip install tensorboard")
    except Exception as e:
        print(f"Error loading TensorBoard logs: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained humanoid walking model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--vec-normalize", type=str, default=None, 
                       help="Path to VecNormalize statistics file")
    parser.add_argument("--episodes", type=int, default=10, 
                       help="Number of episodes to evaluate")
    parser.add_argument("--render", action="store_true", 
                       help="Render the environment during evaluation")
    parser.add_argument("--xml", type=str, default="assets/humanoid_180_75.xml",
                       help="Path to humanoid XML file")
    parser.add_argument("--plot-logs", type=str, default=None,
                       help="Path to TensorBoard logs to plot")
    
    args = parser.parse_args()
    
    # Run evaluation
    print(f"\n🤖 Humanoid Walking Evaluation Tool")
    print("="*60)
    
    summary, metrics = evaluate_model(
        model_path=args.model,
        vec_normalize_path=args.vec_normalize,
        num_episodes=args.episodes,
        render=args.render,
        xml_path=args.xml,
        verbose=True
    )
    
    # Plot training curves if requested
    if args.plot_logs:
        print("\nGenerating training plots...")
        plot_training_curves(args.plot_logs)
