"""
Test overlay function với best MSE samples
"""
import sys
sys.path.insert(0, '.')

from plot_prediction_comparison import plot_overlay_comparison

results_dir = 'results'
output_step = 5
models = ['conv1d_gru', 'gru', 'conv1d']
output_dir = 'test_output'
num_samples = 10

print("=" * 80)
print(f"Testing overlay comparison với {num_samples} best samples")
print("=" * 80)

plot_overlay_comparison(results_dir, output_step, models, output_dir, num_samples=num_samples)

print("\n" + "=" * 80)
print("DONE! Check test_output/predictions_comparison/overlay_out5.png")
print("=" * 80)
