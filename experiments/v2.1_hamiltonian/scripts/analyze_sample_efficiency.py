"""
Phase A.2: Sample Efficiency Analysis

Compare learning speed: Hamiltonian PPO vs Baseline PPO
Metrics:
- Steps to convergence
- Learning curve comparison  
- Sample efficiency ratio
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

print('=' * 70)
print('Phase A.2: Sample Efficiency Analysis')
print('=' * 70)

# Load evaluation data from all configs
exp_root = Path(__file__).parent.parent

configs = [
    ('Baseline (λ=0.0)', 'logs/baseline_100k/evaluations.npz'),
    ('Weak (λ=0.1)', 'logs/hamiltonian_lambda0.1/evaluations.npz'),
    ('Medium (λ=0.5)', 'logs/hamiltonian_lambda0.5/evaluations.npz'),
    ('Strong (λ=1.0)', 'logs/hamiltonian_lambda1.0/evaluations.npz'),
]

all_data = {}

print('\nLoading evaluation data...')
for name, eval_path in configs:
    full_path = exp_root / eval_path
    
    if not full_path.exists():
        print(f'⚠️  Missing: {name} ({eval_path})')
        continue
    
    data = np.load(full_path)
    
    # Extract learning curve
    timesteps = data['timesteps']
    results = data['results']
    ep_lengths = data['ep_lengths']
    
    # Compute mean reward per evaluation
    mean_rewards = [np.mean(r) for r in results]
    
    all_data[name] = {
        'timesteps': timesteps,
        'mean_rewards': mean_rewards,
        'ep_lengths': ep_lengths,
        'final_reward': mean_rewards[-1] if len(mean_rewards) > 0 else None
    }
    
    print(f'✅ Loaded: {name}')
    print(f'   Evaluations: {len(timesteps)}')
    print(f'   Final reward: {mean_rewards[-1]:.2f}')

print('\n' + '=' * 70)
print('Sample Efficiency Analysis')
print('=' * 70)

# Define convergence threshold (90% of best final performance)
best_final = max([d['final_reward'] for d in all_data.values() if d['final_reward'] is not None])
threshold = 0.9 * best_final

print(f'\nBest final performance: {best_final:.2f}')
print(f'Convergence threshold (90%): {threshold:.2f}')

# Compute steps to convergence for each config
convergence_results = {}

for name, data in all_data.items():
    timesteps = np.array(data['timesteps'])
    rewards = np.array(data['mean_rewards'])
    
    # Find first time reward exceeds threshold
    above_threshold = rewards >= threshold
    
    if np.any(above_threshold):
        first_idx = np.argmax(above_threshold)
        steps_to_converge = timesteps[first_idx]
        converged = True
    else:
        steps_to_converge = None
        converged = False
    
    convergence_results[name] = {
        'converged': converged,
        'steps': steps_to_converge,
        'final_reward': data['final_reward']
    }
    
    if converged:
        print(f'\n{name}:')
        print(f'  Converged: ✅ Yes')
        print(f'  Steps to 90%: {steps_to_converge:,}')
    else:
        print(f'\n{name}:')
        print(f'  Converged: ❌ No (did not reach {threshold:.2f})')

# Compute sample efficiency ratio (vs baseline)
baseline_steps = convergence_results.get('Baseline (λ=0.0)', {}).get('steps', None)

if baseline_steps:
    print('\n' + '=' * 70)
    print('Sample Efficiency vs Baseline')
    print('=' * 70)
    
    for name, result in convergence_results.items():
        if name == 'Baseline (λ=0.0)':
            continue
        
        if result['converged'] and result['steps']:
            speedup = baseline_steps / result['steps']
            improvement = (1 - result['steps'] / baseline_steps) * 100
            
            print(f'\n{name}:')
            print(f'  Speedup: {speedup:.2f}×')
            print(f'  Sample efficiency: {improvement:+.1f}%')
            
            if speedup > 1.5:
                print(f'  ✅ Significantly faster')
            elif speedup > 1.1:
                print(f'  ⚠️ Moderately faster')
            else:
                print(f'  ⏸️ Similar speed')

# Visualization
print('\n' + '=' * 70)
print('Generating visualizations...')
print('=' * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Learning curves
ax = axes[0, 0]
colors = {'Baseline (λ=0.0)': 'blue', 'Weak (λ=0.1)': 'orange', 
          'Medium (λ=0.5)': 'green', 'Strong (λ=1.0)': 'red'}

for name, data in all_data.items():
    ax.plot(data['timesteps'], data['mean_rewards'], 
            label=name, color=colors.get(name, 'gray'), linewidth=2, alpha=0.8)

ax.axhline(y=threshold, color='black', linestyle='--', alpha=0.5, 
           label=f'Threshold (90% of best)')
ax.set_xlabel('Training Steps')
ax.set_ylabel('Mean Evaluation Reward')
ax.set_title('Learning Curves Comparison')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

# Plot 2: Steps to convergence
ax = axes[0, 1]
names = []
steps = []
for name, result in convergence_results.items():
    if result['converged']:
        names.append(name.replace(' (λ=', '\n(λ='))
        steps.append(result['steps'])

if len(names) > 0:
    bars = ax.bar(range(len(names)), steps, color=['blue', 'orange', 'green', 'red'][:len(names)])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('Steps to Convergence')
    ax.set_title('Sample Efficiency (Steps to 90% Performance)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, step) in enumerate(zip(bars, steps)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(step/1000)}k',
                ha='center', va='bottom', fontsize=10)

# Plot 3: Speedup vs Baseline
ax = axes[1, 0]
if baseline_steps:
    speedup_names = []
    speedup_values = []
    
    for name, result in convergence_results.items():
        if name != 'Baseline (λ=0.0)' and result['converged'] and result['steps']:
            speedup_names.append(name.replace(' (λ=', '\n(λ='))
            speedup_values.append(baseline_steps / result['steps'])
    
    if len(speedup_names) > 0:
        bars = ax.bar(range(len(speedup_names)), speedup_values, 
                      color=['orange', 'green', 'red'][:len(speedup_names)])
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Baseline')
        ax.set_xticks(range(len(speedup_names)))
        ax.set_xticklabels(speedup_names, fontsize=9)
        ax.set_ylabel('Speedup vs Baseline')
        ax.set_title('Sample Efficiency Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, speedup_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}×',
                    ha='center', va='bottom', fontsize=10)

# Plot 4: Summary table
ax = axes[1, 1]
ax.axis('off')

summary_text = f"""
Sample Efficiency Analysis Summary
{'=' * 45}

Convergence Threshold: {threshold:.2f}
(90% of best performance: {best_final:.2f})

Results:
"""

for name, result in convergence_results.items():
    lambda_val = name.split('λ=')[1].rstrip(')')
    if result['converged']:
        steps_k = result['steps'] / 1000
        summary_text += f"\nλ={lambda_val}: {steps_k:.0f}k steps"
        
        if baseline_steps and name != 'Baseline (λ=0.0)':
            speedup = baseline_steps / result['steps']
            summary_text += f" ({speedup:.2f}× faster)"
    else:
        summary_text += f"\nλ={lambda_val}: Did not converge"

# Add interpretation
if baseline_steps:
    best_speedup_name = None
    best_speedup = 0
    for name, result in convergence_results.items():
        if name != 'Baseline (λ=0.0)' and result['converged'] and result['steps']:
            speedup = baseline_steps / result['steps']
            if speedup > best_speedup:
                best_speedup = speedup
                best_speedup_name = name
    
    if best_speedup_name:
        summary_text += f"\n\n{'=' * 45}\n"
        summary_text += f"Best: {best_speedup_name}\n"
        summary_text += f"Speedup: {best_speedup:.2f}×\n"
        
        if best_speedup > 1.5:
            summary_text += "\n✅ Significant sample efficiency gain"
        elif best_speedup > 1.1:
            summary_text += "\n⚠️ Moderate sample efficiency gain"
        else:
            summary_text += "\n⏸️ Minimal sample efficiency difference"

ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
        fontfamily='monospace', verticalalignment='top', fontsize=9)

plt.tight_layout()
plt.savefig(exp_root / 'analysis/sample_efficiency.png', dpi=150)
print('✅ Saved: analysis/sample_efficiency.png')

# Save detailed results
output_file = exp_root / 'analysis/SAMPLE_EFFICIENCY_RESULTS.md'
with open(output_file, 'w') as f:
    f.write('# Phase A.2: Sample Efficiency Analysis Results\n\n')
    f.write('**Date:** 2026-03-23\n')
    f.write('**Configurations:** Baseline + 3 Hamiltonian variants\n\n')
    
    f.write('## Convergence Threshold\n\n')
    f.write(f'**Best final performance:** {best_final:.2f}\n')
    f.write(f'**90% threshold:** {threshold:.2f}\n\n')
    
    f.write('## Steps to Convergence\n\n')
    f.write('| Configuration | Converged | Steps | vs Baseline |\n')
    f.write('|---------------|-----------|-------|-------------|\n')
    
    for name, result in convergence_results.items():
        if result['converged']:
            steps_str = f"{result['steps']:,}"
            
            if baseline_steps and name != 'Baseline (λ=0.0)':
                speedup = baseline_steps / result['steps']
                improvement = (1 - result['steps'] / baseline_steps) * 100
                vs_baseline = f"{speedup:.2f}× ({improvement:+.1f}%)"
            else:
                vs_baseline = "-"
            
            f.write(f'| {name} | ✅ Yes | {steps_str} | {vs_baseline} |\n')
        else:
            f.write(f'| {name} | ❌ No | - | - |\n')
    
    f.write('\n## Interpretation\n\n')
    
    if baseline_steps:
        best_name = max(
            [(name, baseline_steps / result['steps']) 
             for name, result in convergence_results.items()
             if name != 'Baseline (λ=0.0)' and result['converged'] and result['steps']],
            key=lambda x: x[1],
            default=(None, 0)
        )
        
        if best_name[0]:
            f.write(f'**Best sample efficiency:** {best_name[0]}\n')
            f.write(f'**Speedup:** {best_name[1]:.2f}×\n\n')
            
            if best_name[1] > 1.5:
                f.write('✅ **Significant improvement** in sample efficiency.\n\n')
                f.write('Hamiltonian guidance accelerates learning substantially, ')
                f.write('suggesting physics-informed gradients provide valuable signal ')
                f.write('beyond pure reward-driven exploration.\n\n')
            elif best_name[1] > 1.1:
                f.write('⚠️ **Moderate improvement** in sample efficiency.\n\n')
                f.write('Hamiltonian guidance provides some acceleration, but effect is modest. ')
                f.write('May be task-dependent or require further tuning.\n\n')
            else:
                f.write('⏸️ **Minimal difference** in sample efficiency.\n\n')
                f.write('Hamiltonian guidance does not substantially accelerate learning. ')
                f.write('Primary benefit is final performance, not learning speed.\n\n')
    
    f.write('## Next Steps\n\n')
    f.write('- Share results with 小P for physics interpretation\n')
    f.write('- Proceed to Phase A.3: Parameter sweep\n')
    f.write('- If speedup significant: Investigate mechanism (gradient alignment?)\n')

print(f'✅ Saved: {output_file}')

print('\n' + '=' * 70)
print('Phase A.2 Complete!')
print('=' * 70)

if baseline_steps:
    best = max(
        [(name, baseline_steps / result['steps']) 
         for name, result in convergence_results.items()
         if name != 'Baseline (λ=0.0)' and result['converged'] and result['steps']],
        key=lambda x: x[1],
        default=(None, 0)
    )
    
    if best[0]:
        print(f'\nBest sample efficiency: {best[0]}')
        print(f'Speedup: {best[1]:.2f}×')
        
        if best[1] > 1.5:
            print('Status: ✅ Significant improvement')
        elif best[1] > 1.1:
            print('Status: ⚠️ Moderate improvement')
        else:
            print('Status: ⏸️ Minimal difference')

print(f'\nFiles:')
print(f'  - analysis/sample_efficiency.png')
print(f'  - analysis/SAMPLE_EFFICIENCY_RESULTS.md')
print(f'\n小P review needed: Sample efficiency interpretation')
