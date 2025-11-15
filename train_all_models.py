"""
Script để train tất cả các models (Deep Learning + Baseline)
Chạy tuần tự từng model và lưu kết quả vào thư mục riêng
"""

import os
import sys
import subprocess
from datetime import datetime

# Danh sách models cần train
MODELS = [
    # Deep Learning models
    {
        'name': 'Conv1D-GRU (ResNet)',
        'type': 'conv1d_gru',
        'output_dir': 'results/conv1d_gru',
        'description': 'Best model - Hybrid CNN-RNN with Residual Network'
    },
    {
        'name': 'GRU',
        'type': 'gru',
        'output_dir': 'results/gru',
        'description': 'Pure RNN model with 3 GRU layers'
    },
    {
        'name': 'Conv1D',
        'type': 'conv1d',
        'output_dir': 'results/conv1d',
        'description': 'Pure CNN model for time series'
    },
]

# Baseline models (optional - comment out if don't want to train)
BASELINE_MODELS = [
    {
        'name': 'Linear Regression',
        'type': 'linear',
        'output_dir': 'results/linear',
        'description': 'Simple baseline model'
    },
    {
        'name': 'XGBoost',
        'type': 'xgboost',
        'output_dir': 'results/xgboost',
        'description': 'Tree-based gradient boosting'
    },
    {
        'name': 'LightGBM',
        'type': 'lightgbm',
        'output_dir': 'results/lightgbm',
        'description': 'Fast gradient boosting'
    },
]

# Training parameters
EPOCHS = 1000
BATCH_SIZE = 64
PATIENCE = 10
ADD_NOISE = True


def print_header(title):
    """In header đẹp"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def train_model(model_config, epochs=EPOCHS, batch_size=BATCH_SIZE,
                patience=PATIENCE, add_noise=ADD_NOISE):
    """
    Train một model

    Args:
        model_config: Dict chứa thông tin model
        epochs, batch_size, patience, add_noise: Training parameters

    Returns:
        bool: True nếu thành công, False nếu thất bại
    """
    model_name = model_config['name']
    model_type = model_config['type']
    output_dir = model_config['output_dir']

    print_header(f"TRAINING: {model_name}")
    print(f"Type: {model_type}")
    print(f"Description: {model_config['description']}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}\n")

    # Build command
    cmd = [
        sys.executable,  # Python interpreter
        'main.py',
        '--model_type', model_type,
        '--output_dir', output_dir,
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--patience', str(patience),
    ]

    if add_noise:
        cmd.append('--add_noise')

    # Run training
    start_time = datetime.now()

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)

        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()

        print(f"\n✅ {model_name} trained successfully!")
        print(f"   Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ {model_name} training FAILED!")
        print(f"   Error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  {model_name} training INTERRUPTED by user!")
        raise
    except Exception as e:
        print(f"\n❌ {model_name} training FAILED with unexpected error!")
        print(f"   Error: {e}")
        return False


def main():
    """Main function"""
    print_header("TRAIN ALL MODELS")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total models: {len(MODELS) + len(BASELINE_MODELS)}")
    print(f"  - Deep Learning: {len(MODELS)}")
    print(f"  - Baseline: {len(BASELINE_MODELS)}")

    results = {}

    # Train Deep Learning models
    print_header("PHASE 1: DEEP LEARNING MODELS")
    for model_config in MODELS:
        success = train_model(model_config)
        results[model_config['name']] = success

    # Train Baseline models
    print_header("PHASE 2: BASELINE MODELS")
    for model_config in BASELINE_MODELS:
        success = train_model(model_config)
        results[model_config['name']] = success

    # Print summary
    print_header("TRAINING SUMMARY")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    successful = sum(1 for v in results.values() if v)
    failed = len(results) - successful

    print(f"{'Model':<25} {'Status':<10}")
    print("-" * 40)
    for model_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{model_name:<25} {status}")

    print("\n" + "=" * 40)
    print(f"Total: {len(results)} models")
    print(f"Success: {successful}")
    print(f"Failed: {failed}")
    print("=" * 40)

    if failed == 0:
        print("\n🎉 All models trained successfully!")
    else:
        print(f"\n⚠️  {failed} model(s) failed. Check logs above.")

    return failed == 0


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        sys.exit(130)
