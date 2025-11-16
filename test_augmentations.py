"""
Test script for new augmentation strategies
Demo các augmentation strategies mới theo đề xuất của reviewer
"""

import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import DataPreprocessor
from config import Config

def create_sample_signal(length=100):
    """Tạo sample signal để test"""
    t = np.linspace(0, 4*np.pi, length)
    signal = np.sin(t) + 0.5 * np.sin(3*t) + 0.2 * np.random.randn(length)
    return signal

def test_multiple_noise_levels():
    """Test với nhiều mức độ noise khác nhau"""
    print("=" * 80)
    print("TEST 1: MULTIPLE NOISE LEVELS")
    print("=" * 80)

    signal = create_sample_signal()
    noise_factors = [0.05, 0.1, 0.15, 0.2]

    fig, axes = plt.subplots(len(noise_factors) + 1, 1, figsize=(12, 10))

    # Original signal
    axes[0].plot(signal, 'b-', linewidth=2, label='Original Signal')
    axes[0].set_title('Original Signal', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Noisy signals
    for idx, noise_factor in enumerate(noise_factors):
        noisy_signal = DataPreprocessor.add_noise(signal, noise_level_factor=noise_factor)
        axes[idx + 1].plot(signal, 'b-', alpha=0.3, label='Original')
        axes[idx + 1].plot(noisy_signal, 'r-', linewidth=1, label=f'Noise factor={noise_factor}')
        axes[idx + 1].set_title(f'Gaussian Noise (σ = {noise_factor} × std)', fontweight='bold')
        axes[idx + 1].legend()
        axes[idx + 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('test_multiple_noise_levels.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: test_multiple_noise_levels.png\n")

def test_random_dropout():
    """Test random dropout augmentation"""
    print("=" * 80)
    print("TEST 2: RANDOM DROPOUT OF SEGMENTS")
    print("=" * 80)

    signal = create_sample_signal()

    # Test với các dropout probabilities khác nhau
    dropout_probs = [0.05, 0.1, 0.15]

    fig, axes = plt.subplots(len(dropout_probs) + 1, 1, figsize=(12, 10))

    # Original signal
    axes[0].plot(signal, 'b-', linewidth=2, label='Original Signal')
    axes[0].set_title('Original Signal', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Dropout signals
    for idx, dropout_prob in enumerate(dropout_probs):
        dropout_signal = DataPreprocessor.add_random_dropout(
            signal,
            dropout_prob=dropout_prob,
            min_length=1,
            max_length=5
        )
        axes[idx + 1].plot(signal, 'b-', alpha=0.3, label='Original')
        axes[idx + 1].plot(dropout_signal, 'g-', linewidth=1.5, label=f'Dropout prob={dropout_prob}')
        axes[idx + 1].set_title(f'Random Dropout (p = {dropout_prob})', fontweight='bold')
        axes[idx + 1].legend()
        axes[idx + 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('test_random_dropout.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: test_random_dropout.png\n")

def test_block_missingness():
    """Test block missingness augmentation"""
    print("=" * 80)
    print("TEST 3: BLOCK MISSINGNESS")
    print("=" * 80)

    signal = create_sample_signal()

    # Test với các fill methods khác nhau
    fill_methods = ['zero', 'mean', 'interpolate']

    fig, axes = plt.subplots(len(fill_methods) + 1, 1, figsize=(12, 10))

    # Original signal
    axes[0].plot(signal, 'b-', linewidth=2, label='Original Signal')
    axes[0].set_title('Original Signal', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Block missingness signals
    for idx, fill_method in enumerate(fill_methods):
        block_signal = DataPreprocessor.add_block_missingness(
            signal,
            block_prob=1.0,  # Force block missingness để demo
            min_length=5,
            max_length=15,
            fill_method=fill_method
        )
        axes[idx + 1].plot(signal, 'b-', alpha=0.3, label='Original')
        axes[idx + 1].plot(block_signal, 'm-', linewidth=1.5, label=f'Fill: {fill_method}')
        axes[idx + 1].set_title(f'Block Missingness (fill={fill_method})', fontweight='bold')
        axes[idx + 1].legend()
        axes[idx + 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('test_block_missingness.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: test_block_missingness.png\n")

def test_combined_strategies():
    """Test tất cả strategies combined"""
    print("=" * 80)
    print("TEST 4: COMBINED AUGMENTATION STRATEGIES")
    print("=" * 80)

    signal = create_sample_signal()

    # Áp dụng tất cả strategies
    strategies = ['noise', 'dropout', 'block_missingness']

    augmented_datasets = DataPreprocessor.apply_augmentations(
        signal,
        strategies=strategies,
        noise_factors=[0.1],
        use_multiple_noise=False
    )

    fig, axes = plt.subplots(len(augmented_datasets) + 1, 1, figsize=(12, 10))

    # Original signal
    axes[0].plot(signal, 'b-', linewidth=2, label='Original Signal')
    axes[0].set_title('Original Signal', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Augmented signals
    strategy_names = ['Gaussian Noise', 'Random Dropout', 'Block Missingness']
    colors = ['r', 'g', 'm']

    for idx, (aug_data, strategy_name, color) in enumerate(zip(augmented_datasets, strategy_names, colors)):
        axes[idx + 1].plot(signal, 'b-', alpha=0.3, label='Original')
        axes[idx + 1].plot(aug_data, f'{color}-', linewidth=1.5, label=strategy_name)
        axes[idx + 1].set_title(strategy_name, fontweight='bold')
        axes[idx + 1].legend()
        axes[idx + 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('test_combined_strategies.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: test_combined_strategies.png\n")

def compare_all_strategies():
    """So sánh tất cả strategies trên cùng 1 plot"""
    print("=" * 80)
    print("TEST 5: COMPARISON OF ALL STRATEGIES")
    print("=" * 80)

    signal = create_sample_signal()

    # Tạo augmented versions
    noisy = DataPreprocessor.add_noise(signal, noise_level_factor=0.1)
    dropout = DataPreprocessor.add_random_dropout(signal, dropout_prob=0.1)
    block = DataPreprocessor.add_block_missingness(signal, block_prob=1.0, fill_method='interpolate')

    # Plot comparison
    plt.figure(figsize=(14, 8))

    plt.plot(signal, 'b-', linewidth=2, label='Original Signal', alpha=0.7)
    plt.plot(noisy, 'r-', linewidth=1, label='Gaussian Noise (σ=0.1)', alpha=0.6)
    plt.plot(dropout, 'g-', linewidth=1, label='Random Dropout (p=0.1)', alpha=0.6)
    plt.plot(block, 'm-', linewidth=1, label='Block Missingness', alpha=0.6)

    plt.title('Comparison of All Augmentation Strategies', fontsize=16, fontweight='bold')
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig('test_comparison_all.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: test_comparison_all.png\n")

def main():
    """Main test function"""
    print("\n" + "=" * 80)
    print("  AUGMENTATION STRATEGIES TEST SUITE")
    print("  Response to Reviewer Comments on Robustness Testing")
    print("=" * 80 + "\n")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Run tests
    test_multiple_noise_levels()
    test_random_dropout()
    test_block_missingness()
    test_combined_strategies()
    compare_all_strategies()

    print("\n" + "=" * 80)
    print("✅ ALL TESTS COMPLETED!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  ✓ test_multiple_noise_levels.png")
    print("  ✓ test_random_dropout.png")
    print("  ✓ test_block_missingness.png")
    print("  ✓ test_combined_strategies.png")
    print("  ✓ test_comparison_all.png")
    print("\nSee AUGMENTATION_GUIDE.md for detailed documentation.")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user!")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
