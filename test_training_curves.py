"""
Test script để vẽ training curves comparison
"""
import os
import pickle
import matplotlib.pyplot as plt

results_dir = 'results'
output_dir = 'test_output'
os.makedirs(output_dir, exist_ok=True)

# Test với output_step = 5
out_step = 5
models = ['conv1d_gru', 'gru', 'conv1d']

print(f"Testing training curves for output_step={out_step}")
print("=" * 60)

# Colors
colors = {
    'conv1d_gru': '#2ecc71',
    'gru': '#3498db',
    'conv1d': '#e74c3c'
}

# Load histories
histories = {}

for model in models:
    history_file = os.path.join(results_dir, str(out_step), model, 'history_saved.pkl')

    print(f"\nModel: {model}")
    print(f"  File: {history_file}")
    print(f"  Exists: {os.path.exists(history_file)}")

    if os.path.exists(history_file):
        try:
            with open(history_file, 'rb') as f:
                history = pickle.load(f)
            histories[model] = history

            print(f"  Keys: {list(history.keys())}")
            print(f"  Epochs: {len(history.get('loss', []))}")
            print(f"  OK Loaded successfully!")

        except Exception as e:
            print(f"  ERROR: {e}")

if not histories:
    print("\nERROR: No histories loaded!")
    exit(1)

# Plot
print(f"\n{'=' * 60}")
print("Creating plot...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Training Loss
ax1 = axes[0]
for model, history in histories.items():
    if 'loss' in history:
        epochs = range(1, len(history['loss']) + 1)
        ax1.plot(epochs, history['loss'],
                linewidth=2, label=model.upper().replace('_', '-'),
                color=colors.get(model, '#95a5a6'), alpha=0.8)

ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax1.set_title(f'Training Loss - Output Steps = {out_step}',
             fontsize=14, fontweight='bold', pad=15)
ax1.legend(fontsize=11, loc='best')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_yscale('log')

# Validation Loss
ax2 = axes[1]
for model, history in histories.items():
    if 'val_loss' in history:
        epochs = range(1, len(history['val_loss']) + 1)
        ax2.plot(epochs, history['val_loss'],
                linewidth=2, label=model.upper().replace('_', '-'),
                color=colors.get(model, '#95a5a6'), alpha=0.8)

ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
ax2.set_title(f'Validation Loss - Output Steps = {out_step}',
             fontsize=14, fontweight='bold', pad=15)
ax2.legend(fontsize=11, loc='best')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_yscale('log')

plt.tight_layout()

# Save
output_file = os.path.join(output_dir, f'training_curves_out{out_step}.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"OK Saved: {output_file}")
print(f"\n{'=' * 60}")
print("SUCCESS!")
