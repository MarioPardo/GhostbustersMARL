#!/usr/bin/env python3
"""
Evaluate a trained QMIX/VDN policy with visualization.

Usage:
    python evaluate_policy.py \
        --checkpoint results/models/ghostbusters_qmix_seed665699370_ghostbusters_stage2_2025-11-19\ 11:12:28.055738 \
        --config ghostbusters_qmix \
        --env-config ghostbusters_stage2 \
        --n-episodes 10
"""

import sys
import os

# Add pymarl/src to path
pymarl_src = os.path.join(os.path.dirname(__file__), 'pymarl', 'src')
sys.path.insert(0, pymarl_src)

from main import ex

if __name__ == '__main__':
    # Parse custom args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to model checkpoint directory')
    parser.add_argument('--config', type=str, default='ghostbusters_qmix',
                       help='Algorithm config name')
    parser.add_argument('--env-config', type=str, default='ghostbusters_stage2',
                       help='Environment config name')
    parser.add_argument('--n-episodes', type=int, default=10,
                       help='Number of episodes to run')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable rendering')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args, unknown = parser.parse_known_args()
    
    # Build sacred command line args (Sacred uses 'with' keyword for config updates)
    sacred_args = [
        'with',
        f'name={args.config}',
        f'config={args.config}',
        f'env_config={args.env_config}',
        f'checkpoint_path={args.checkpoint}',
        'evaluate=True',
        f'render={not args.no_render}',  # Enable by default
        'render_every=1',  # Render EVERY frame during evaluation
        f'test_nepisode={args.n_episodes}',
        f'seed={args.seed}',
        'save_replay=False',
        't_max=0',  # No training
    ]
    
    # Run with sacred
    ex.run_commandline(sacred_args)
